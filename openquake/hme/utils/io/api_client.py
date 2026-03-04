"""
API client for loading flatfile data from the Ground Motion API server.

This module provides a drop-in replacement for load_flatfile() that queries
a remote API instead of loading from CSV files.
"""

import logging
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Exception raised for API client errors"""

    pass


def _create_session_with_retries(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: tuple = (500, 502, 504),
) -> requests.Session:
    """
    Create a requests session with retry logic.

    Args:
        retries: Number of retries
        backoff_factor: Backoff factor for retries
        status_forcelist: HTTP status codes to retry on

    Returns:
        Configured requests session
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def compute_rupture_bbox(
    rupture_gdf: gpd.GeoDataFrame, buffer_degrees: float = 2.0
) -> Tuple[float, float, float, float]:
    """
    Compute bounding box from rupture locations with buffer.

    Handles International Date Line crossing by detecting when max_lon < min_lon
    and normalizing coordinates.

    Args:
        rupture_gdf: GeoDataFrame of ruptures
        buffer_degrees: Buffer to add around bbox in degrees

    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat)

    Note:
        For IDL crossing regions, returns bbox in [0, 360] range to indicate
        wrapping. Caller should handle this appropriately.
    """
    bounds = rupture_gdf.total_bounds  # [minx, miny, maxx, maxy]

    min_lon = bounds[0]
    max_lon = bounds[2]
    min_lat = bounds[1]
    max_lat = bounds[3]

    # Detect International Date Line crossing
    # If max_lon < min_lon, the region crosses ±180°
    crosses_idl = max_lon < min_lon

    if crosses_idl:
        logger.warning(
            "Rupture region crosses International Date Line. "
            f"Raw bounds: [{min_lon:.2f}, {max_lon:.2f}]. "
            "Converting to [0, 360] range for bbox query."
        )
        # Convert negative longitudes to [0, 360] range
        # This allows proper bbox computation across IDL
        lons = []
        for geom in rupture_gdf.geometry:
            lon = geom.x
            if lon < 0:
                lon += 360.0
            lons.append(lon)

        min_lon = min(lons) - buffer_degrees
        max_lon = max(lons) + buffer_degrees

        # Keep in [0, 360] range
        min_lon = max(min_lon, 0.0)
        max_lon = min(max_lon, 360.0)

        # Convert back to [-180, 180] for API compatibility
        # Split into two ranges if still crossing after buffer
        if max_lon > 180.0:
            # Still crosses IDL even after normalization
            # Return special marker: min_lon > max_lon indicates split query needed
            min_lon_a = min_lon - 360.0  # Western part
            max_lon_a = 180.0
            min_lon_b = -180.0
            max_lon_b = max_lon - 360.0  # Eastern part

            logger.warning(
                f"Bbox still crosses IDL after buffer. Will need split query: "
                f"[{min_lon_a:.2f}, {max_lon_a:.2f}] + [{min_lon_b:.2f}, {max_lon_b:.2f}]"
            )

            # For now, return expanded bbox that covers both sides
            # This may over-query but ensures we don't miss data
            return (
                -180.0,
                min_lat - buffer_degrees,
                180.0,
                max_lat + buffer_degrees,
            )
        else:
            # Convert back to [-180, 180]
            if min_lon > 180.0:
                min_lon -= 360.0
            if max_lon > 180.0:
                max_lon -= 360.0
    else:
        # Normal case: no IDL crossing
        min_lon -= buffer_degrees
        max_lon += buffer_degrees

    # Clip to valid lat/lon ranges
    min_lon = max(min_lon, -180.0)
    max_lon = min(max_lon, 180.0)
    min_lat = max(min_lat - buffer_degrees, -90.0)
    max_lat = min(max_lat + buffer_degrees, 90.0)

    return min_lon, min_lat, max_lon, max_lat


def load_flatfile_from_api(
    base_url: str,
    min_mag: Optional[float] = None,
    max_mag: Optional[float] = None,
    h3_res: Optional[int] = None,
    rupture_gdf: Optional[gpd.GeoDataFrame] = None,
    buffer_degrees: float = 2.0,
    timeout: int = 300,
    prefer_h3_filter: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load flatfile data from API server.

    This is a drop-in replacement for load_flatfile() that queries a remote
    API instead of loading from CSV. It uses spatial filtering based on
    rupture locations to minimize data transfer and memory usage.

    Strategy:
        1. If ruptures have H3 cells and prefer_h3_filter=True:
           Use H3 cell list for exact filtering (best for res 2-5)
        2. Otherwise: Use bounding box spatial query (works with any resolution)

    Args:
        base_url: Base URL of the API (e.g., "http://localhost:8000/api/v1")
        min_mag: Minimum magnitude filter
        max_mag: Maximum magnitude filter
        h3_res: H3 resolution for spatial indexing
        rupture_gdf: GeoDataFrame of ruptures for spatial filtering
        buffer_degrees: Buffer around rupture bbox in degrees (bbox method only)
        timeout: Request timeout in seconds
        prefer_h3_filter: If True and ruptures have cell_id, use H3 filtering

    Returns:
        Tuple of (eq_gm_df, gm_df) matching load_flatfile() output

    Raises:
        APIClientError: If API requests fail
    """
    logger.info(f"Loading flatfile from API: {base_url}")

    session = _create_session_with_retries()

    # Step 1: Determine filtering strategy
    use_h3_filter = False
    cells_in_model = None
    h3_col_name = None

    if rupture_gdf is not None and len(rupture_gdf) > 0 and prefer_h3_filter:
        # Check for pre-computed H3 cells at the requested resolution
        if h3_res is not None:
            h3_col_name = f"cell_id_res{h3_res}"

            if h3_col_name in rupture_gdf.columns:
                # Use pre-computed H3 cells at exact resolution
                cells_in_model = rupture_gdf[h3_col_name].unique().tolist()
                logger.info(
                    f"Found pre-computed H3 cells at resolution {h3_res} "
                    f"(column: {h3_col_name})"
                )
            elif "cell_id" in rupture_gdf.columns:
                # Fall back to generic cell_id column (backward compatibility)
                cells_in_model = rupture_gdf.cell_id.unique().tolist()
                h3_col_name = "cell_id"
                logger.info(
                    f"Using generic 'cell_id' column "
                    f"(resolution may not match {h3_res})"
                )

        if (
            cells_in_model
            and len(cells_in_model) > 0
            and len(cells_in_model) < 10000
        ):
            use_h3_filter = True
            logger.info(
                f"Using H3 cell filtering with {len(cells_in_model)} cells "
                f"at resolution {h3_res} (exact matching, no over-query)"
            )
        elif cells_in_model and len(cells_in_model) >= 10000:
            logger.warning(
                f"Too many H3 cells ({len(cells_in_model)}) for direct query, "
                f"falling back to bbox filtering"
            )
            use_h3_filter = False

    # Step 2: Query earthquakes with spatial and magnitude filters
    params = {"limit": 100_000}  # Large limit for bulk loading

    if min_mag is not None:
        params["min_mag"] = min_mag
        logger.info(f"Filtering min_mag >= {min_mag}")

    if max_mag is not None:
        params["max_mag"] = max_mag
        logger.info(f"Filtering max_mag <= {max_mag}")

    # Apply spatial filtering
    if use_h3_filter:
        # H3 cell filtering (exact, no IDL issues)
        params["cell_ids"] = ",".join(cells_in_model)
        params["h3_res"] = h3_res  # Tell server which resolution to use
        logger.info(
            f"Querying {len(cells_in_model)} H3 cells at resolution {h3_res} "
            f"using pre-computed column: {h3_col_name}"
        )
    elif rupture_gdf is not None and len(rupture_gdf) > 0:
        # Bounding box filtering (works with any resolution)
        min_lon, min_lat, max_lon, max_lat = compute_rupture_bbox(
            rupture_gdf, buffer_degrees
        )
        bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        params["bbox"] = bbox_str
        logger.info(
            f"Filtering by rupture bbox with {buffer_degrees}° buffer: "
            f"[{min_lon:.2f}, {min_lat:.2f}] to [{max_lon:.2f}, {max_lat:.2f}]"
        )

    try:
        response = session.get(
            f"{base_url}/earthquakes", params=params, timeout=timeout
        )
        response.raise_for_status()
        eq_data = response.json()
    except requests.HTTPError as e:
        # Print the response body to see validation details
        logging.warning(f"Response body: {e.response.text}")
        raise APIClientError(f"Failed to fetch earthquakes: {e}")

    if not eq_data.get("earthquakes"):
        logger.warning("No earthquakes returned from API")
        return pd.DataFrame(), pd.DataFrame()

    eq_gm_df = pd.DataFrame(eq_data["earthquakes"])
    logger.info(f"Loaded {len(eq_gm_df)} earthquakes from API")

    # Step 2: Query ground motions in bulk for all earthquakes
    event_ids = eq_gm_df["event_id"].tolist()

    if not event_ids:
        logger.warning("No earthquakes to query ground motions for")
        return eq_gm_df, pd.DataFrame()

    try:
        response = session.post(
            f"{base_url}/ground-motions/bulk",
            json={"event_ids": event_ids},
            timeout=timeout,
        )
        response.raise_for_status()
        gm_data = response.json()
    except requests.exceptions.RequestException as e:
        raise APIClientError(f"Failed to fetch ground motions: {e}")

    gm_df = pd.DataFrame(gm_data["ground_motions"])
    logger.info(f"Loaded {len(gm_df)} ground motion records from API")

    # Step 3: Add H3 cell IDs if requested
    if h3_res is not None:
        try:
            import h3

            logger.info(f"Computing H3 cells at resolution {h3_res}")

            # Add H3 cells to earthquakes
            eq_gm_df["cell_id"] = [
                h3.latlng_to_cell(row.latitude, row.longitude, h3_res)
                for _, row in eq_gm_df.iterrows()
            ]

            # Add H3 cells to ground motions
            if len(gm_df) > 0:
                gm_df["cell_id"] = [
                    h3.latlng_to_cell(row.st_latitude, row.st_longitude, h3_res)
                    for _, row in gm_df.iterrows()
                ]
        except ImportError:
            logger.warning(
                "h3 library not available, skipping H3 cell computation"
            )

    # Step 4: Add geometry columns (matching load_flatfile behavior)
    if len(eq_gm_df) > 0:
        eq_gm_df["geometry"] = eq_gm_df.apply(
            lambda row: Point(row.longitude, row.latitude, row.depth), axis=1
        )

    if len(gm_df) > 0:
        gm_df["geometry"] = gm_df.apply(
            lambda row: Point(row.st_longitude, row.st_latitude), axis=1
        )

    logger.info("Flatfile loading from API complete")
    return eq_gm_df, gm_df


def load_flatfile_from_api_with_cells(
    base_url: str,
    min_mag: Optional[float] = None,
    max_mag: Optional[float] = None,
    h3_res: Optional[int] = None,
    cell_ids: Optional[List[str]] = None,
    timeout: int = 300,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load flatfile data from API using H3 cell filtering.

    This function is useful when you already have H3 cells computed at the
    desired resolution and want to filter directly by cell IDs.

    Args:
        base_url: Base URL of the API
        min_mag: Minimum magnitude filter
        max_mag: Maximum magnitude filter
        h3_res: H3 resolution (for validation)
        cell_ids: List of H3 cell IDs to filter by
        timeout: Request timeout in seconds

    Returns:
        Tuple of (eq_gm_df, gm_df)

    Raises:
        APIClientError: If API requests fail
    """
    logger.info(
        f"Loading flatfile from API with {len(cell_ids or [])} H3 cells"
    )

    session = _create_session_with_retries()

    # Note: This requires the API to support cell_id filtering
    # For now, we'll query all and filter client-side
    # TODO: Add cell_ids parameter to API endpoint

    params = {"limit": 100_000}
    if min_mag is not None:
        params["min_mag"] = min_mag
    if max_mag is not None:
        params["max_mag"] = max_mag

    try:
        response = session.get(
            f"{base_url}/earthquakes", params=params, timeout=timeout
        )
        response.raise_for_status()
        eq_data = response.json()
    except requests.exceptions.RequestException as e:
        raise APIClientError(f"Failed to fetch earthquakes: {e}")

    eq_gm_df = pd.DataFrame(eq_data["earthquakes"])

    # Filter by cell_ids if provided
    if cell_ids and "cell_id" in eq_gm_df.columns:
        eq_gm_df = eq_gm_df[eq_gm_df.cell_id.isin(cell_ids)]
        logger.info(
            f"Filtered to {len(eq_gm_df)} earthquakes in specified cells"
        )

    # Get ground motions
    event_ids = eq_gm_df["event_id"].tolist()
    if not event_ids:
        return eq_gm_df, pd.DataFrame()

    try:
        response = session.post(
            f"{base_url}/ground-motions/bulk",
            json={"event_ids": event_ids},
            timeout=timeout,
        )
        response.raise_for_status()
        gm_data = response.json()
    except requests.exceptions.RequestException as e:
        raise APIClientError(f"Failed to fetch ground motions: {e}")

    gm_df = pd.DataFrame(gm_data["ground_motions"])

    # Add geometry columns
    if len(eq_gm_df) > 0:
        eq_gm_df["geometry"] = eq_gm_df.apply(
            lambda row: Point(row.longitude, row.latitude, row.depth), axis=1
        )

    if len(gm_df) > 0:
        gm_df["geometry"] = gm_df.apply(
            lambda row: Point(row.st_longitude, row.st_latitude), axis=1
        )

    return eq_gm_df, gm_df
