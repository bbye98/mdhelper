"""
Analysis base classes
=====================
.. moduleauthor:: Benjamin B. Ye <bye@caltech.edu>

This module contains custom base classes :class:`SerialAnalysisBase` and
:class:`ParallelAnalysisBase` for serial and multithreaded data 
analysis, respectively, with the latter supporting the native 
multiprocessing, Dask, and Joblib libraries for parallelization.
"""

from abc import abstractmethod
from datetime import datetime
import multiprocessing
import os
from typing import TextIO, Union
import warnings

try:
    import dask
    from dask import distributed
    FOUND_DASK = True
except ImportError:
    FOUND_DASK = False

try:
    import joblib
    FOUND_JOBLIB = True
except ImportError:
    FOUND_JOBLIB = False

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.coordinates.base import ReaderBase
import numpy as np

from .. import ArrayLike
from ..utility import log

class SerialAnalysisBase(AnalysisBase):

    """
    A serial analysis base object.

    Parameters
    ----------
    trajectory : `MDAnalysis.coordinates.base.ReaderBase`
        Simulation trajectory.
    
    verbose : `bool`, default: :code:`True`
        Determines whether detailed progress is shown.

    **kwargs
        Additional keyword arguments to pass to
        :class:`MDAnalysis.analysis.base.AnalysisBase`.
    """

    def __init__(
            self, trajectory: ReaderBase, verbose: bool = False, **kwargs):
        super().__init__(trajectory, verbose, **kwargs)

    def run(
            self, start: int = None, stop: int = None, step: int = None,
            frames: Union[slice, ArrayLike] = None, verbose: bool = None, 
            **kwargs) -> "SerialAnalysisBase":
        
        """
        Perform the calculation in serial.

        Parameters
        ----------
        start : `int`, optional
            Starting frame for analysis.
        
        stop : `int`, optional
            Ending frame for analysis.

        step : `int`, optional
            Number of frames to skip between each analyzed frame.

        frames : `slice` or array-like, optional
            Index or logical array of the desired trajectory frames.

        verbose : `bool`, optional
            Determines whether detailed progress is shown.
        
        **kwargs
            Additional keyword arguments to pass to
            :class:`MDAnalysis.lib.log.ProgressBar`.

        Returns
        -------
        self : `SerialAnalysisBase`
            Serial analysis base object.
        """

        return super().run(start, stop, step, frames, verbose, **kwargs)

    def save(
            self, file: Union[str, TextIO], archive: bool = True,
            compress: bool = True, **kwargs) -> None:

        """
        Saves results to a binary or archive file in NumPy format.

        Parameters
        ----------
        file : `str` or `file`
            Filename or file-like object where the data will be saved.
            If `file` is a `str`, the :code:`.npy` or :code:`.npz` 
            extension will be appended automatically if not already
            present.
        
        archive : `bool`, default: :code:`True`
            Determines whether the results are saved to a single archive
            file. If `True`, the data is stored in a :code:`.npz` file.
            Otherwise, the data is saved to multiple :code:`.npy` files.

        compress : `bool`, default: :code:`True`
            Determines whether the :code:`.npz` file is compressed. Has
            no effect when :code:`archive=False`.

        **kwargs
            Additional keyword arguments to pass to :func:`numpy.save`,
            :func:`numpy.savez`, or :func:`numpy.savez_compressed`,
            depending on the values of `archive` and `compress`.
        """

        if archive and compress:
            np.savez_compressed(file, **self.results, **kwargs)
        elif archive:
            np.savez(file, **self.results, **kwargs)
        else:
            for data in self.results:
                np.save(f"{file}_{data}", self.results[data], **kwargs)

class ParallelAnalysisBase(SerialAnalysisBase):

    """
    A multithreaded analysis base object.

    Parameters
    ----------
    trajectory : `MDAnalysis.coordinates.base.ReaderBase`
        Simulation trajectory.
    
    verbose : `bool`, default: :code:`True`
        Determines whether detailed progress is shown.

    **kwargs
        Additional keyword arguments to pass to
        :class:`MDAnalysis.analysis.base.AnalysisBase`.
    """

    def __init__(
            self, trajectory: ReaderBase, verbose: bool = False, **kwargs):
        super().__init__(trajectory, verbose, **kwargs)

    def _dask_job_block(
            self, frames: Union[slice, np.ndarray], indices: np.ndarray) -> list:
        return [self._single_frame(f, i) for f, i in zip(frames, indices)]

    @abstractmethod
    def _single_frame(self, frame: int, index: int):
        pass

    def run(
            self, start: int = None, stop: int = None, step: int = None,
            frames: Union[slice, ArrayLike] = None, verbose: bool = None,
            n_jobs: int = None, module: str = "joblib", block: bool = True,
            method: str = None, **kwargs) -> "ParallelAnalysisBase":

        """
        Perform the calculation in parallel.

        Parameters
        ----------
        start : `int`, optional
            Starting frame for analysis.
        
        stop : `int`, optional
            Ending frame for analysis.

        step : `int`, optional
            Number of frames to skip between each analyzed frame.

        frames : `slice` or array-like, optional
            Index or logical array of the desired trajectory frames.

        verbose : `bool`, optional
            Determines whether detailed progress is shown.

        n_jobs : `int`, keyword-only, optional
            Number of workers. If not specified, it is automatically
            set to either the minimum number of workers required to
            fully analyze the trajectory or the maximum number of CPU
            threads available.

        module : `str`, keyword-only, default: :code:`"joblib"`
            Parallelization module to use for analysis.

            **Valid values**: :code:`"dask"`, :code:`"joblib"`, and 
            :code:`"multiprocessing"`.

        block : `bool`, keyword-only, default: :code:`True`
            Determines whether the trajectory is split into smaller
            blocks that are processed serially in parallel with other
            blocks. This "split–apply–combine" approach is generally
            faster since the trajectory attributes do not have to be
            packaged for each analysis run. Has no effect if
            :code:`module="multiprocessing"`.
        
        method : `str`, keyword-only, optional
            Specifies which Dask scheduler, Joblib backend, or
            multiprocessing start method is used.
        
        **kwargs
            Additional keyword arguments to pass to
            :func:`dask.compute`, :class:`joblib.Parallel`, or
            :class:`multiprocessing.pool.Pool`, depending on the value of
            `module`.

        Returns
        -------
        self : `ParallelAnalysisBase`
            Parallel analysis base object.
        """

        verbose = getattr(self, '_verbose', False) if verbose is None \
                  else verbose

        self._setup_frames(self._trajectory, start=start, stop=stop,
                           step=step, frames=frames)
        self._prepare()

        n_jobs = min(n_jobs or np.inf, self.n_frames, 
                     len(os.sched_getaffinity(0)))
        frames = frames if frames \
                 else np.arange(self.start, self.stop, self.step)
        indices = np.arange(len(frames))

        if module == "dask" and FOUND_DASK:
            try:
                config = {
                    "scheduler": distributed.worker.get_client(),
                    **kwargs
                }
                n_jobs = min(len(config["scheduler"].get_worker_logs()), 
                             n_jobs)
            except ValueError:
                if method is None:
                    method = "processes"
                elif method not in {"distributed", "processes", "threading",
                                  "threads", "single-threaded", "sync",
                                  "synchronous"}:
                    raise ValueError("Invalid Dask scheduler.")
                    
                if method == "distributed":
                    raise RuntimeError("The Dask distributed client "
                                       "(client = dask.distributed.Client(...)) "
                                       "should be instantiated in the main "
                                       "program (__name__ = '__main__') of "
                                       "your script.")
                elif method in {"threading", "threads"}:
                    raise ValueError("The threaded Dask scheduler is not "
                                     "compatible with MDAnalysis.")
                elif n_jobs == 1 and method not in {"single-threaded", "sync",
                                                    "synchronous"}:
                    method = "synchronous"
                    warnings.warn(f"Since {n_jobs=}, the synchronous "
                                  "Dask scheduler will be used instead.")
                config = {"scheduler": method} | kwargs
                if method == "processes":
                    config["num_workers"] = n_jobs

            if verbose:
                log(f"Starting analysis using Dask ({n_jobs=}, "
                    f"scheduler={config['scheduler']})...")

            jobs = []
            if block:
                for frame, index in zip(
                        np.array_split(frames, n_jobs), 
                        np.array_split(indices, n_jobs)
                    ):
                    jobs.append(dask.delayed(self._dask_job_block)(frame, index))
            else:
                for frame, index in zip(frames, indices):
                    jobs.append(dask.delayed(self._single_frame)(frame, index))

            if verbose:
                time_start = datetime.now()

            self._results = dask.delayed(jobs).compute(**config)
            if block:
                self._results = [r for b in self._results for r in b]

        elif module == "joblib" and FOUND_JOBLIB:
            if method is not None and method not in {"processes", "threads", 
                                                     None}:
                raise ValueError("Invalid Joblib backend.")

            if verbose:
                log("Starting analysis using Joblib "
                    f"({n_jobs=}, backend={method})...")
                time_start = datetime.now()
            
            if block:
                self._results = joblib.Parallel(n_jobs=n_jobs, prefer=method, 
                                                **kwargs)(
                    joblib.delayed(self._single_frame)(f, i)
                    for frames_, indices_ in zip(
                        np.array_split(frames, n_jobs),
                        np.array_split(indices, n_jobs)
                    ) for f, i in zip(frames_, indices_)
                )
            else:
                self._results = joblib.Parallel(n_jobs=n_jobs, prefer=method, 
                                                **kwargs)(
                    joblib.delayed(self._single_frame)(f, i) 
                    for f, i in zip(frames, indices)
                )

        else:
            if module != "multiprocessing":
                warnings.warn("The Dask or Joblib library was not "
                              "found, so the native multiprocessing "
                              "module will be used instead.")
            
            if method is None:
                method = multiprocessing.get_start_method()
            elif method not in {"fork", "forkserver", "spawn"}:
                raise ValueError("Invalid multiprocessing start method.")

            if verbose:
                log("Starting analysis using multiprocessing "
                    f"({n_jobs=}, {method=})...")
                time_start = datetime.now()

            with multiprocessing.get_context(method).Pool(n_jobs, **kwargs) as p:
                self._results = p.starmap(self._single_frame, 
                                          zip(frames, indices))
        if verbose:
            log(f"Finished! Time elapsed: {datetime.now() - time_start}.")

        self._conclude()
        return self