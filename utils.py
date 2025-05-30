import logging
import warnings

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NoDataBlockWarning(Warning):
    pass

def get_extremes_block_maxima(
    ts: pd.Series,
    extremes_type: Literal["high", "low"],
    block_size: Union[str, pd.Timedelta] = "365.2425D",
    errors: Literal["raise", "ignore", "coerce"] = "raise",
    min_last_block: Optional[float] = None,
) -> pd.Series:
    """
    Optimized version of get_extremes_block_maxima for faster block maxima extraction.
    """

    logger.debug(
        "collecting block maxima extreme events using extremes_type=%s, "
        "block_size=%s, errors=%s",
        extremes_type,
        block_size,
        errors,
    )

    if extremes_type not in ["high", "low"]:
        raise ValueError(
            f"invalid value in '{extremes_type}' for the 'extremes_type' argument"
        )

    if errors not in ["raise", "ignore", "coerce"]:
        raise ValueError(f"invalid value in '{errors}' for the 'errors' argument")

    if not isinstance(block_size, pd.Timedelta):
        if isinstance(block_size, str):
            block_size = pd.to_timedelta(block_size)
        else:
            raise TypeError(
                f"invalid type in {type(block_size).__name__} for 'block_size'"
            )

    # Convert datetime index to integer timestamps (nanoseconds)
    ts = ts.sort_index()
    start = ts.index[0]
    delta_ns = block_size.to_timedelta64().astype('timedelta64[ns]').astype(np.int64)

    timestamps = ts.index.view(np.int64)
    block_ids = (timestamps - timestamps[0]) // delta_ns

    # Group by block ids using numpy for performance
    df = pd.DataFrame({'block': block_ids, 'value': ts.values, 'timestamp': ts.index})

    # Use groupby with custom aggregation to avoid multiple passes
    if extremes_type == "high":
        idx = df.groupby("block")["value"].idxmax()
    else:
        idx = df.groupby("block")["value"].idxmin()

    result = df.loc[idx, ["timestamp", "value"]].set_index("timestamp").squeeze()

    # Handle potential empty blocks
    total_blocks = (timestamps[-1] - timestamps[0]) // delta_ns + 1
    all_blocks = np.arange(total_blocks)
    present_blocks = df["block"].unique()
    missing_blocks = set(all_blocks) - set(present_blocks)

    empty_intervals = len(missing_blocks)
    if empty_intervals > 0:
        for mb in sorted(missing_blocks):
            left = pd.to_datetime(timestamps[0] + mb * delta_ns)
            right = left + block_size
            mid = left + (right - left) / 2
            if errors == "coerce":
                result.loc[mid] = np.nan
            elif errors == "ignore":
                continue
            else:
                raise ValueError(
                    f"no data in block [{left} ; {right}), "
                    f"fill gaps in the data or set 'errors' to 'coerce' or 'ignore'"
                )

        if errors == "coerce":
            result = result.sort_index().fillna(result.mean())

        # warnings.warn(
        #     message=f"{empty_intervals} blocks contained no data",
        #     category=NoDataBlockWarning,
        # )

    # Check last block duration
    if min_last_block is not None:
        last_block_start = timestamps[0] + (total_blocks - 1) * delta_ns
        last_block_end = timestamps[0] + total_blocks * delta_ns
        ratio = (timestamps[-1] - last_block_start) / delta_ns
        if ratio < min_last_block:
            last_mid = pd.to_datetime(last_block_start + delta_ns // 2)
            if last_mid in result.index:
                result = result.drop(last_mid)
            logger.debug("discarded last block with data availability ratio of %s", ratio)

    result.name = ts.name or "extreme values"
    result.index.name = ts.index.name or "date-time"

    logger.debug(
        "successfully collected %d extreme events, found %s no-data blocks",
        len(result),
        empty_intervals,
    )

    return result.astype(np.float64)