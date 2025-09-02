"""
Dual Problem Interface for Quadratically Constrained Quadratic Programming (QCQP).

This module provides interfaces for solving QCQP problems with shared projection 
constraints using dual optimization methods. It includes both sparse and dense 
implementations optimized for different matrix structures.
"""

__all__ = ["SparseSharedProjQCQP", "DenseSharedProjQCQP"]

import copy
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import (
    Any,
    Iterator,
    Optional,
    Tuple,
    cast,
)

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sksparse.cholmod
from numpy.typing import ArrayLike

from dolphindes.types import ComplexArray, FloatNDArray
from dolphindes.util import CRdot, Sym

from .optimization import BFGS, Alt_Newton_GD, _Optimizer


class _SharedProjQCQP(ABC):
    """Represents a QCQP with a single type of constraint over projection regions.

    Parent class, should not be instantiated directly as it is missing
    key functionality.

    Problem is
    max_x -x^dagger A0 x + 2 Re (x^dagger s0) + c0
    s.t.  Re(-x^dagger A1 P_j A_2 x) + 2 Re (x^dagger A_2^dagger P_j^dagger s1) = 0

    for a list of projection matrices P_j.

    Attributes
    ----------
    A0 : scipy.sparse.csc_array | np.ndarray
        The matrix A0 in the QCQP.
    s0 : np.ndarray
        The vector s in the QCQP.
    c0 : float
        The constant c in the QCQP.
    A1 : scipy.sparse.csc_array | np.ndarray
        The matrix A1 in the QCQP.
    A2 : scipy.sparse.csc_array | np.ndarray
        The matrix A2 in the QCQP.
    s1 : np.ndarray
        The vector s1 in the QCQP.
    Pdiags : np.ndarray
        The diagonal elements of the projection matrices P_j, as columns
        of a matrix Pdiags. The second column j should always be one such
        that A0 + lambda A1 P_j is positive semidefinite for sufficiently
        large constant lambda.
    verbose : int
        The verbosity level for debugging and logging.
    Acho : sksparse.cholmod.Cholesky | np.ndarray
        The Cholesky factorization of the total A matrix, which is updated
        when needed.
    current_dual : float
        The current dual solution, which is only updated when the dual
        problem is solved.
    current_lags : np.ndarray
        The current Lagrangian multipliers, which is only updated when
        the dual problem is solved.
    current_grad : np.ndarray
        The current grad_lambda of the dual solution, which is only
        updated when the dual problem is solved.
    current_hess : np.ndarray
        The current hess_lambda of the dual solution, which is only
        updated when the dual problem is solved.
    """

    def __init__(
        self,
        A0: ArrayLike | sp.csc_array,
        s0: ArrayLike,
        c0: float,
        A1: ArrayLike | sp.csc_array,
        A2: ArrayLike | sp.csc_array,
        s1: ArrayLike,
        Pdiags: ArrayLike,
        verbose: int = 0,
    ) -> None:
        # Cast A0 to appropriate type
        if sp.issparse(A0):
            self.A0 = sp.csc_array(A0)
        else:
            self.A0 = cast(ComplexArray, np.asarray(A0, dtype=complex))
        
        # Cast A1 to appropriate type
        if sp.issparse(A1):
            self.A1 = sp.csc_array(A1)
        else:
            self.A1 = cast(ComplexArray, np.asarray(A1, dtype=complex))
        
        # Cast A2 to appropriate type
        if sp.issparse(A2):
            self.A2 = sp.csc_array(A2)
        else:
            self.A2 = cast(ComplexArray, np.asarray(A2, dtype=complex))
        
        # Cast vectors to ComplexArray
        self.s0 = cast(ComplexArray, np.asarray(s0, dtype=complex))
        self.s1 = cast(ComplexArray, np.asarray(s1, dtype=complex))
        self.Pdiags = cast(ComplexArray, np.asarray(Pdiags, dtype=complex))
        
        self.c0 = c0
        self.verbose = verbose
        self.Acho: Optional[Any] = None
        self.current_dual: Optional[float] = None
        self.current_grad: Optional[FloatNDArray] = None
        self.current_hess: Optional[FloatNDArray] = None
        self.current_lags: Optional[FloatNDArray] = None
        self.current_xstar: Optional[ComplexArray] = None
        self.use_precomp = True

        if self.use_precomp:
            self.compute_precomputed_values()

    def compute_precomputed_values(self) -> None:
        """
        Precompute the A constraint matrices for each projector.

        This makes _get_total_A faster for a small number of constraints.
        Also precompute Fs = A2^dagger P_k^dagger s1 for each projector,
        which makes _get_total_S faster.
        """
        self.precomputed_As = []
        for i in range(self.Pdiags.shape[1]):
            Ak = Sym(
                self.A1 @ sp.diags_array(self.Pdiags[:, i], format="csr") @ self.A2
            )
            self.precomputed_As.append(Ak)

        if self.verbose > 0:
            print(
                f"Precomputed {self.Pdiags.shape[1]} A matrices for "
                "the projectors."
            )

        # (Fs)_k = A_2^dagger P_k^dagger s1
        self.Fs = self.A2.conj().T @ (self.Pdiags.conj().T * self.s1).T

    def get_number_constraints(self) -> int:
        """Return the number of constraints in the QCQP."""
        return self.Pdiags.shape[1]

    def _add_projectors(self, lags: FloatNDArray) -> ComplexArray:
        """Combine the lagrange multipliers and the projectors into a joint projector.

        Parameters
        ----------
        lags : np.ndarray
            The lagrange multipliers for the projectors.

        Returns
        -------
        np.ndarray
            The diagonal elements of the combined projector matrix sum_j lag[j] P_j
        """
        return cast(ComplexArray, self.Pdiags @ lags)

    def _get_total_A(self, lags: FloatNDArray) -> sp.csc_array | ComplexArray:
        """Get the total A matrix for the QCQP = A0 + sum_j lag[j] A1 P_j A2."""
        return (
            self._get_total_A_precomp(lags)
            if self.use_precomp
            else self._get_total_A_noprecomp(lags)
        )

    def _get_total_A_precomp(self, lags: FloatNDArray) -> sp.csc_array | ComplexArray:
        """Get the total A matrix using precomputed values.
        
        Should be faster for fewer constraints.
        """
        return self.A0 + sum(
            lags[i] * self.precomputed_As[i] for i in range(len(lags))
        )

    def _get_total_A_noprecomp(self, lags: FloatNDArray) -> sp.csc_array | ComplexArray:
        """Get the total A matrix without using precomputed_As.
        
        Should be faster for many constraints.
        """
        # TODO(alessio): P_diag is usually already computed before calling
        # get_total_A. Keeping it like this to not have to change the passed
        # arguments, but should fix it at some point.
        P_diag = self._add_projectors(lags)
        return self.A0 + Sym(self.A1 @ sp.diags_array(P_diag, format="csr") @ self.A2)

    def _get_total_S(self, Pdiag: ComplexArray) -> ComplexArray:
        """Get the total S vector for the QCQP.
        
        Returns s0 + sum_j lag[j] P_j^dagger s1 given P = sum_j lag[j] P_j
        """
        return cast(ComplexArray, self.s0 + self.A2.T.conj() @ (Pdiag.conj() * self.s1))

    @abstractmethod
    def _update_Acho(self, A: sp.csc_array | ComplexArray) -> None:
        """
        Update the Cholesky factorization to be that of the input matrix A.

        Needs to be implemented by subclasses.

        Parameters
        ----------
        A : sp.csc_array or np.ndarray
            New total A.
        """
        pass

    @abstractmethod
    def _Acho_solve(self, b: ComplexArray) -> ComplexArray:
        """
        Compute A^{-1} b using Acho.

        Needs to be implemented by subclasses.

        Parameters
        ----------
        b : np.ndarray
            Right-hand-side of linear system.

        Returns
        -------
        np.ndarray
            A^{-1} b
        """
        pass

    @abstractmethod
    def is_dual_feasible(self, lags: FloatNDArray) -> bool:
        """
        Check if a set of Lagrange multipliers is dual feasible.

        Needs to be implemented by subclasses.

        Arguments
        ---------
        lags: np.ndarray
            A 1-dimensional numpy array of Lagrange multipliers.

        Returns
        -------
        bool
            True if the total A matrix is positive semidefinite (dual feasible).
        """
        pass

    def find_feasible_lags(
        self, start: float = 0.1, limit: float = 1e8
    ) -> FloatNDArray:
        """
        Find a feasible point for the dual problem.

        This assumes that for large enough lags[1], A is PSD.

        Parameters
        ----------
        start : float (optional, default 1.0)
            The start value for lags[1]
        limit : float (optional, default 1e8)
            A maximum value for lags[1] before the method gives up and raises an error.

        Returns
        -------
        init_lags : np.ndarray
            A set of Lagrange multipliers such that A is PSD.
        """
        if (self.current_lags is not None) and self.is_dual_feasible(self.current_lags):
            return self.current_lags

        # Start with small positive lags
        init_lags = np.random.random(int(self.Pdiags.shape[1])) * 1e-6
        
        init_lags[1] = start
        while self.is_dual_feasible(init_lags) is False:
            init_lags[1] *= 1.5
            if init_lags[1] > limit:
                raise ValueError(
                    "Could not find a feasible point for the dual problem."
                )

        if self.verbose > 0:
            print(
                f"Found feasible point for dual problem: {init_lags} with "
                f"dualvalue {self.get_dual(init_lags)[0]}"
            )
        return init_lags

    def _get_PSD_penalty(self, lags: FloatNDArray) -> Tuple[ComplexArray, float]:
        """
        Return the eigenvector for the smallest eigenvalue.

        This vector can be used to calculate the PSD boundary of the dual function.
        In theory, other PSD penalties exist, such as the determinant. We found that
        this works well for photonic limits.

        Parameters
        ----------
        lags : np.ndarray
            Lagrange multipliers from which A will be calculated

        Returns
        -------
        eigv : np.ndarray
            Eigenvector corresponding to the lowest eigenvalue of A(lags)
        eigw : float
            Eigenvalue corresponding to the lowest eigenvalue of A(lags)
        """
        A = self._get_total_A(lags)
        eigw, eigv = spla.eigsh(A, k=1, sigma=0.0, which="LM", return_eigenvectors=True)
        return eigv[:, 0], eigw[0]

    def _get_xstar(self, lags: FloatNDArray) -> Tuple[ComplexArray, float]:
        """Solve for xstar using the cholesky factorization of A.

        For a given set of Lagrange multipliers, which define A and S, the x_star
        that maximizes the Lagrangian (and therefore defines the dual) is given by
        A x_star = S. Since the symbolic Cholesky factorization of A is known,
        because the sparsity pattern of A is always known, this solution can be
        found very efficiently.

        Parameters
        ----------
        lags : np.ndarray
            The lagrange multipliers for the projectors.

        Returns
        -------
        x_star : np.ndarray
            The solution x_star to the equation A x_star = S.
        xAx : float
            x_star.conj().T @ A @ x_star, the dual function value
        """
        P_diag = self._add_projectors(lags)
        A = self._get_total_A(lags)
        S = self._get_total_S(P_diag)
        self._update_Acho(A)  # update the Cholesky factorization

        x_star: ComplexArray = self._Acho_solve(S)
        xAx: float = np.real(np.vdot(x_star, A @ x_star))

        return x_star, xAx

    def get_dual(
        self,
        lags: FloatNDArray,
        get_grad: bool = False,
        get_hess: bool = False,
        penalty_vectors: Optional[list[FloatNDArray]] = None,
    ) -> Tuple[float, Optional[FloatNDArray], Optional[FloatNDArray], Any]:
        """Get the dual value and derivatives with respect to Lagrange multipliers.

        Parameters
        ----------
        lags : np.ndarray
            The lagrange multipliers for the projectors.
        get_grad : bool, optional
            Whether to return the gradient of the dual with respect to the
            lagrange multipliers. Default is False.
        get_hess : bool, optional
            Whether to return the Hessian of the dual with respect to the
            lagrange multipliers. Default is False. If True, grad is also
            automatically computed regardless of what get_grad specifies.
        penalty_vectors : list, optional
            A list of penalty vectors for the PSD boundary. If provided, the
            dual value and gradients will include a penalty term. Default is None.

        Returns
        -------
        dualval : float
            The value of the dual function.
        grad : np.ndarray or None
            The gradient of the dual function with respect to the lagrange
            multipliers, if requested.
        hess : np.ndarray or None
            The Hessian of the dual function with respect to the lagrange
            multipliers, if requested.
        dual_aux : namedtuple
            Auxiliary information about the dual computation.
        """
        if penalty_vectors is None:
            penalty_vectors = []

        grad, hess = None, None
        grad_penalty, hess_penalty = np.array([]), np.array([[]])

        xstar, dualval = self._get_xstar(lags)
        dualval += self.c0

        if get_hess:
            if not hasattr(self, "precomputed_As"):
                raise AttributeError("precomputed_As needed for computing Hessian")
                # this assumes that in the future we may consider making
                # precomputed_As optional can also compute the Hessian without
                # precomputed_As, leave for future implementation if useful

            # useful intermediate computations
            # (Fx)_k = -Sym(A_1 P_k A2) x_star = -A_k @ x_star

            Fx = np.zeros((len(xstar), len(self.precomputed_As)), dtype=complex)
            for k, Ak in enumerate(self.precomputed_As):
                Fx[:, k] = -Ak @ xstar

            # get_hess implies get_grad also
            grad = np.real(xstar.conj() @ (Fx + 2 * self.Fs))

            Ftot = Fx + self.Fs
            hess = 2 * np.real(Ftot.conj().T @ self._Acho_solve(Ftot))

        elif get_grad: 
            # This is grad_lambda (not grad_x); elif since get_hess computes grad
            # First term: -Re(xstar.conj() @ self.A1 @ (self.Pdiags[:, i] *
            # (self.A2 @ xstar))). Second term: 2*Re(xstar.conj() @
            # self.A2.T.conj() @ (self.Pdiags[:, i].conj() * self.s1))
            # self.Pdiags has shape (N_diag, N_projectors), A2_xstar has shape (N_diag,)
            # We want to multiply each column of Pdiags elementwise with A2_xstar.
            # However, we know that sum_i w_i A_ij v_i = sum_i (w_i * v_i) A_ij.
            # LHS is expression right below, RHS is below so we avoid dense
            # intermediate matrices.
            A2_xstar = self.A2 @ xstar  # Shape: (N_p,)
            # Shape: (N_p,) where N_p = self.A1.shape[1]
            x_conj_A1 = (xstar.conj() @ self.A1)  
            
            # term1 = -np.real((xstar.conj() @ self.A1) @ (self.Pdiags *
            # A2_xstar[:, np.newaxis]))  # Shape: (N_diag, N_projectors)
            term1 = -np.real((x_conj_A1 * A2_xstar) @ self.Pdiags)

            # term2 = 2 * np.real(A2_xstar.conj() @ (self.Pdiags.conj() *
            # self.s1[:, np.newaxis])) #. Same as above.
            term2 = 2 * np.real((A2_xstar.conj() * self.s1) @ self.Pdiags.conj())
            grad = term1 + term2

        # Boundary penalty for the PSD boundary
        dualval_penalty = 0.0
        if len(penalty_vectors) > 0:
            penalty_matrix = np.column_stack(penalty_vectors).astype(complex)
            A_inv_penalty = self._Acho_solve(penalty_matrix)
            dualval_penalty += np.sum(
                np.real(A_inv_penalty.conj() * penalty_matrix)
            )  # multiplies columns with columns, sums all at once

            if get_hess:
                grad = cast(FloatNDArray, grad)
                penalty_matrix = cast(ComplexArray, penalty_matrix)
                # get_hess implies get_grad also
                grad_penalty = np.zeros(grad.shape[0])
                hess_penalty = np.zeros((grad.shape[0], grad.shape[0]))
                Fv = np.zeros((penalty_matrix.shape[0], len(grad)), dtype=complex)
                for j in range(penalty_matrix.shape[1]):
                    for k, Ak in enumerate(self.precomputed_As):
                        # yes this is a double for loop, hessian for fake sources
                        # is likely a speed bottleneck
                        Fv[:, k] = Ak @ A_inv_penalty[:, j]

                    grad_penalty += np.real(-A_inv_penalty[:, j].conj().T @ Fv)
                    hess_penalty += 2 * np.real(Fv.conj().T @ self._Acho_solve(Fv))

            elif get_grad:
                grad = cast(FloatNDArray, grad)
                grad_penalty = np.zeros(grad.shape[0])
                for j in range(penalty_matrix.shape[1]):
                    # slow: for i in range(len(grad)): grad_penalty[i] +=
                    # -np.real(A_inv_penalty[:, j].conj().T @ self.A1 @
                    # (self.Pdiags[:, i] * (self.A2 @ A_inv_penalty[:, j])))
                    # # for loop method
                    # fast: grad_penalty += -np.real((A_inv_penalty[:, j].conj().T
                    # @ self.A1) @ (self.Pdiags * (self.A2 @
                    # A_inv_penalty[:, j])[:, np.newaxis]))

                    # Same as above (fastest)
                    A_inv_penalty_j_A1 = A_inv_penalty[:, j].conj().T @ self.A1
                    A2_A_inv_penalty_j = self.A2 @ A_inv_penalty[:, j]  # Shape: (N_p,)
                    grad_penalty += -np.real(
                        (A_inv_penalty_j_A1 * A2_A_inv_penalty_j) @ self.Pdiags
                    )

        DualAux = namedtuple(
            "DualAux",
            ["dualval_real", "dualgrad_real", "dualval_penalty", "grad_penalty"],
        )
        dual_aux = DualAux(
            dualval_real=dualval,
            dualgrad_real=grad,
            dualval_penalty=dualval_penalty,
            grad_penalty=grad_penalty,
        )

        if len(penalty_vectors) > 0:
            return (
                dualval + dualval_penalty,
                grad + grad_penalty,
                hess + hess_penalty,
                dual_aux,
            )
        else:
            return dualval, grad, hess, dual_aux

    def solve_current_dual_problem(
        self,
        method: str,
        opt_params: Optional[dict[str, Any]] = None,
        init_lags: Optional[ArrayLike] = None,
    ) -> Tuple[float, ArrayLike, ArrayLike, Optional[ArrayLike], ArrayLike]:
        """
        Solve the current dual problem using the specified method.

        Parameters
        ----------
        method : str
            The method to use for solving the dual problem. 'newton' or 'bfgs'
        opt_params : dict, optional
            Parameters for the optimization algorithm.
        init_lags : np.ndarray, optional
            Initial Lagrange multipliers. If None, will find feasible point
            automatically.

        Returns
        -------
        current_dual : float
            The optimal dual value.
        current_lags : np.ndarray
            The optimal Lagrange multipliers.
        current_grad : np.ndarray
            The gradient at the optimal point.
        current_hess : np.ndarray
            The Hessian at the optimal point.
        current_xstar : np.ndarray
            The dual-optimal primal variable.
        """
        is_convex = True

        OPT_PARAMS_DEFAULTS = {
            "opttol": 1e-2,
            "gradConverge": False,
            "min_inner_iter": 5,
            "max_restart": np.inf,
            "penalty_ratio": 1e-2,
            "penalty_reduction": 0.1,
            "break_iter_period": 20,
            "verbose": self.verbose - 1,
            "penalty_vector_list": [],
        }
        if opt_params is None:
            opt_params = {}
        opt_params = {
            **OPT_PARAMS_DEFAULTS,
            **opt_params,
        }  # override defaults with user specifications

        if init_lags is None:
            init_lags = self.find_feasible_lags()
        init_lags = np.array(init_lags, float)

        optfunc = self.get_dual
        feasibility_func = self.is_dual_feasible
        penalty_vector_func = self._get_PSD_penalty

        optimizer: _Optimizer
        if method == "newton":
            optimizer = Alt_Newton_GD(
                optfunc, feasibility_func, penalty_vector_func, is_convex, opt_params
            )
        elif method == "bfgs":
            optimizer = BFGS(
                optfunc, feasibility_func, penalty_vector_func, is_convex, opt_params
            )
        else:
            raise ValueError(
                f"Unknown method '{method}' for solving the dual problem. "
                "Use newton or bfgs."
            )

        self.current_lags, self.current_dual, self.current_grad, self.current_hess = (
            optimizer.run(init_lags)
        )
        self.current_xstar = self._get_xstar(self.current_lags)[0]

        return (
            self.current_dual,
            self.current_lags,
            self.current_grad,
            self.current_hess,
            self.current_xstar,
        )

    def merge_lead_constraints(self, merged_num: int = 2) -> None:
        """Merge the lead constraints of this QCQP.
        
        See module-level merge_lead_constraints() for details.

        Parameters
        ----------
        merged_num : int, optional
            Number of leading constraints to merge, by default 2.
        """
        merge_lead_constraints(self, merged_num=merged_num)

    def add_constraints(
        self, added_Pdiag_list: list[ComplexArray], orthonormalize: bool = True
    ) -> None:
        """Add new constraints to this QCQP.
        
        See module-level add_constraints() for details.

        Parameters
        ----------
        added_Pdiag_list : list
            List of constraint diagonal matrices to add.
        orthonormalize : bool, optional
            Whether to orthonormalize constraints, by default True.
        """
        add_constraints(
            self, added_Pdiag_list=added_Pdiag_list, orthonormalize=orthonormalize
        )

    def run_gcd(
        self,
        max_cstrt_num: int = 10,
        orthonormalize: bool = True,
        opt_params: Optional[dict[str, Any]] = None,
        max_gcd_iter_num: int = 50,
        gcd_iter_period: int = 5,
        gcd_tol: float = 1e-2,
    ) -> None:
        """Run GCD to approach tightest dual bound for this QCQP.
        
        See module-level run_gcd() for details. Modifies the existing QCQP object.

        Parameters
        ----------
        max_cstrt_num : int
            The maximum number of constraints to consider.
        orthonormalize : bool
            Whether to orthonormalize the constraints.
        opt_params : dict
            Optimization parameters.
        max_gcd_iter_num : int
            Maximum number of GCD iterations.
        gcd_iter_period : int
            Period for GCD iterations.
        gcd_tol : float
            Tolerance for GCD convergence.

        """
        run_gcd(
            self,
            max_cstrt_num=max_cstrt_num,
            orthonormalize=orthonormalize,
            opt_params=opt_params,
            max_gcd_iter_num=max_gcd_iter_num,
            gcd_iter_period=gcd_iter_period,
            gcd_tol=gcd_tol,
        )

class SparseSharedProjQCQP(_SharedProjQCQP):
    """Represents a QCQP with a single type of constraint over projection regions.

    Problem is
    max_x -x^dagger A0 x + 2 Re (x^dagger s0) + c0
    s.t.  Re(-x^dagger A1 P_j A_2 x) + 2 Re (x^dagger A_2^dagger P_j^dagger s1) = 0

    for a list of projection matrices P_j. Representas

    Attributes
    ----------
    A0 : scipy.sparse.csc_array
        The matrix A0 in the QCQP.
    s0 : np.ndarray
        The vector s in the QCQP.
    c0 : float
        The constant c in the QCQP.
    A1 : scipy.sparse.csc_array
        The matrix A1 in the QCQP.
    A2 : scipy.sparse.csc_array
        The matrix A2 in the QCQP.
    s1 : np.ndarray
        The vector s1 in the QCQP.
    Pdiags : np.ndarray
        The diagonal elements of the projection matrices P_j, as columns of a 
        matrix Pdiags. The second column j should always be one such that 
        A0 + lambda A1 P_j is positive semidefinite for sufficiently large 
        constant lambda.
    verbose : float
        The verbosity level for debugging and logging.
    Acho : sksparse.cholmod.Factor
        The Cholesky factorization of the total A matrix, which is updated when 
        needed.
    current_dual : float
        The current dual solution, which is only updated when the dual problem 
        is solved.
    current_lags : np.ndarray
        The current Lagrangian multipliers, which is only updated when the dual 
        problem is solved.
    current_grad : np.ndarray
        The current grad_lambda of the dual solution, which is only updated when 
        the dual problem is solved.
    current_hess : np.ndarray
        The current hess_lambda of the dual solution, which is only updated when 
        the dual problem is solved.
    """

    def __repr__(self) -> str:
        """Return string representation of SparseSharedProjQCQP."""
        return (
            f"SparseSharedProjQCQP of size {self.A0.shape[0]}^2 with "
            f"{self.Pdiags.shape[1]} projectors."
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "SparseSharedProjQCQP":
        """Create a deep copy of the SparseSharedProjQCQP object.
        
        Custom __deepcopy__ because Acho is not pickle-able.

        Parameters
        ----------
        memo : dict
            Memoization dictionary for copy.deepcopy.

        Returns
        -------
        SparseSharedProjQCQP
            A deep copy of this object.
        """
        # custom __deepcopy__ because Acho is not pickle-able
        new_QCQP = SparseSharedProjQCQP.__new__(SparseSharedProjQCQP)
        for name, value in self.__dict__.items():
            if name != "Acho":
                setattr(new_QCQP, name, copy.deepcopy(value, memo))

        new_QCQP._initialize_Acho()  # Recompute the Cholesky factorization. 
        # If dense, will use self.current_lags.
        # TODO: update Acho with current_lags if applicable
        return new_QCQP

    def compute_precomputed_values(self) -> None:
        """Sparse version of compute_precomputed_values."""
        super().compute_precomputed_values()
        self._initialize_Acho()

    def _initialize_Acho(self) -> sksparse.cholmod.Factor:
        """
        Analyze the non-zero structure of A. 
        
        Initialize self.Acho with the optimal fill-reducing permutation for A using 
        sksparse.cholmod
        """
        random_lags = np.random.rand(self.Pdiags.shape[1])
        # P = self._add_projectors(random_lags)
        A = self._get_total_A(random_lags)
        if self.verbose > 1:
            print(
                f"analyzing A of format and shape {type(A)}, {A.shape} "
                f"and # of nonzero elements '{A.count_nonzero()}"
            )
        self.Acho = sksparse.cholmod.analyze(A)

    def _update_Acho(self, A: sp.csc_array) -> None:
        """
        Update the Cholesky factorization to be that of the input matrix A.

        Parameters
        ----------
        A : sp.csc_array
            New total A.
        """
        self.Acho.cholesky_inplace(A)

    def _Acho_solve(self, b):
        """
        Computes A^{-1} b using Acho

        Parameters
        ----------
        b : np.ndarray
            Right-hand-side of linear system.

        Returns
        -------
        A^{-1} b
        """
        return self.Acho.solve_A(b)

    def is_dual_feasible(self, lags: np.ndarray) -> bool:
        """
        Checks if a set of Lagrange multipliers is dual feasible by attempting a Cholesky decomposition.
        This is efficient because the sparsity pattern of the total A matrix is known and does not change.

        Arguments
        ---------
        lags: np.ndarray
            A 1-dimensional numpy array of Lagrange multipliers.

        Returns
        -------
        bool
            True if the total A matrix is positive semidefinite (dual feasible).
        """
        A = self._get_total_A(lags)
        try:
            tmp = self.Acho.cholesky(A)
            tmp = (
                tmp.L()
            )  # Have to access the factor for the decomposition to be actually done.
            return True
        except sksparse.cholmod.CholmodNotPositiveDefiniteError:
            return False

    def refine_projectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Doubles the number of projectors, refining the number of constraints to smaller regions. Multipliers will be selected so that dual value remains constant and can be further optimized from existing point.

        For each projector P_j (columns of Pdiags) that doesn't project to a single pixel, split it into two projectors P1 and P2 such that P_j = P1 + P2 with half (or near half) the nonzero entries; j>2 (zero-th and first projector is always left as is). If there are only two projectors, keep both split projectors and original ones.

        Then, form a new Pdiags with the new projectors. Furthermore, extend lags. For each split projector P_j -> P_j, P_j+1, make lags[j] = lags[j], lags[j+1] = lags[j].

        Updates Pdiags and current_lags attributes. Verifies the dual value remains the same after refinement.

        Arguments
        ---------
        None

        Returns
        -------
        self.Pdiags, self.current_lags
        """
        assert (
            self.current_lags is not None
        ), "Cannot refine projectors until an existing problem is solved. Run solve_current_dual_problem first."
        assert np.all(
            self.Pdiags[:, 0] == 1
        ), "The zeroth projector must contain all ones (identity)"
        assert np.all(
            np.isclose(self.Pdiags[:, 1], -1j)
        ), "The second projector must contain all -1j values (-1j * identity)"

        new_Pdiags_cols = []
        new_lags_list = []

        # Handle the first projectors (j=0,1) - always kept as is
        new_Pdiags_cols.append(self.Pdiags[:, 0])
        new_lags_list.append(self.current_lags[0])

        new_Pdiags_cols.append(self.Pdiags[:, 1])
        new_lags_list.append(self.current_lags[1])

        # Iterate through the rest of the projectors (j > 0)
        split_limit = 0 if self.Pdiags.shape[1] == 2 else 2
        for j in range(split_limit, self.Pdiags.shape[1]):
            P_j_diag = self.Pdiags[:, j]
            current_lag_j = self.current_lags[j]

            # Find non-zero elements (pixels) in the current projector
            # Assuming projector diagonals are binary (0 or 1)
            nonzero_indices = np.where(P_j_diag != 0)[0]
            num_nonzero = len(nonzero_indices)

            if num_nonzero == 1:  # Projector is already a single pixel, keep it as is
                new_Pdiags_cols.append(P_j_diag)
                # new_lags_list.append(current_lag_j)
            elif num_nonzero > 1:  # Split the projector
                split_point = num_nonzero // 2
                indices1 = nonzero_indices[:split_point]
                indices2 = nonzero_indices[split_point:]

                # Create new projector P1
                P1_diag_new = np.zeros_like(P_j_diag)
                P1_diag_new[indices1] = P_j_diag[
                    indices1
                ]  # Copy values from original projector

                # Create new projector P2
                P2_diag_new = np.zeros_like(P_j_diag)
                P2_diag_new[indices2] = P_j_diag[
                    indices2
                ]  # Copy values from original projector

                if not np.all(P1_diag_new[1:] == 0):
                    new_Pdiags_cols.append(P1_diag_new)
                if not np.all(P2_diag_new[1:] == 0):
                    new_Pdiags_cols.append(P2_diag_new)

                if self.verbose > 1:
                    print(
                        f"Split projector {j} (lag: {current_lag_j:.2e}) with {num_nonzero} non-zeros into two projectors with {len(indices1)} and {len(indices2)} non-zeros."
                    )

            else:
                raise ValueError(
                    f"Unexpected number of non-zero elements in projector {j}: {num_nonzero}. Projector should have at least one non-zero element."
                )

        # Update Pdiags and current_lags
        # We need the overall P (and therefore the dual value) to stay constant upon projector refinement.
        # Here, we solve the system of equations to do that. It should always be solvable.
        new_Pdiags = np.column_stack(new_Pdiags_cols)
        new_Pdiags_real = np.vstack(
            [np.real(new_Pdiags), np.imag(new_Pdiags)]
        )  # Convert complex to real for least squares
        old_Pdiags_real = np.vstack(
            [np.real(self.Pdiags), np.imag(self.Pdiags)]
        )  # Convert complex to real for least squares
        self.current_lags, residuals, rank, s = np.linalg.lstsq(
            new_Pdiags_real, old_Pdiags_real @ self.current_lags, rcond=None
        )
        self.Pdiags = new_Pdiags

        # If we've been using the precomputed values, we just added new Pdiags so it may be beneficial to stop doing this
        # TODO(alessio): Debug this, it works but not as intended
        # if self.use_precomp:
        #     self.compute_precomputed_values()  # Get new precomputed values for the comparison in the next run
        #     self.update_use_precomp()

        self.compute_precomputed_values()  # Get new precomputed values for the comparison in the next run

        # Reset current dual, grad, hess, xstar as they are for the old problem structure
        new_dual, new_grad, new_hess, dual_aux = self.get_dual(
            self.current_lags, get_grad=True, get_hess=False
        )
        if self.verbose >= 1:
            print(
                f"previous dual: {self.current_dual}, new dual: {new_dual} (should be the same)"
            )
        assert np.isclose(
            new_dual, self.current_dual, rtol=1e-2
        ), "Dual value should be the same after refinement."

        self.current_dual = new_dual
        self.current_grad = new_grad
        self.current_hess = new_hess
        self.current_xstar = self.current_xstar

        return self.Pdiags, self.current_lags

    def iterative_splitting_step(
        self, method: str = "bfgs", max_cstrt_num: int = np.inf
    ) -> Iterator[Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Iterative splitting step generator function that continues until pixel-level constraints are reached.

        Each call:
            1. Doubles the number of projectors using refine_projectors
            2. Solves the new dual problem and yields the results
            3. Continues until all projectors are "one-hot" matrices (pixel-level constraints)

        Parameters
        ----------
        method : str
            Optimizer used for solving each iterative splitting step
        max_cstrt_num : int or np.inf
            termination condition based on maximum number of constraints.
            If np.inf, defaults to pixel level constraints

        Yields
        ------
        tuple
            Result of solve_current_dual_problem:
            (current_dual, current_lags, current_grad, current_hess, current_xstar)
        """
        max_cstrt_num = min(max_cstrt_num, 2 * self.Pdiags.shape[0])
        # Check if we're already at termination condition
        if self.Pdiags.shape[1] >= max_cstrt_num:
            if self.verbose > 0:
                print("Projector number already above specified max or pixel level.")
            return

        # Continue splitting until number of constraints exceeds or equals max_cstrt_num
        while self.Pdiags.shape[1] < max_cstrt_num:
            if self.verbose > 0:
                if self.Pdiags.shape[1] == 2:
                    print(
                        f"Splitting projectors: {self.Pdiags.shape[1]} → {self.Pdiags.shape[1] + self.Pdiags.shape[1]}"
                    )
                else:
                    print(
                        f"Splitting projectors: {self.Pdiags.shape[1]} → {self.Pdiags.shape[1] + self.Pdiags.shape[1] - 2}"
                    )

            # Refine projectors to get finer constraints
            self.refine_projectors()

            # Solve the dual problem with the new projectors
            result = self.solve_current_dual_problem(
                method, init_lags=self.current_lags
            )

            # Yield the result to the caller
            yield result

            # Check if we've reached pixel level after this iteration
            if self.Pdiags.shape[1] >= max_cstrt_num:
                if self.verbose > 0:
                    print(
                        "Reached max number of projectors or pixel-level projectors. Refinement complete."
                    )
                break


class DenseSharedProjQCQP(_SharedProjQCQP):
    """Represents a QCQP with a single type of constraint over projection regions.

    Problem is
    max_x -x^dagger A0 x + 2 Re (x^dagger s0) + c0
    s.t.  Re(-x^dagger A1 P_j A_2 x) + 2 Re (x^dagger A_2^dagger P_j^dagger s1) = 0

    for a list of projection matrices P_j.

    Attributes
    ----------
    A0 : np.ndarray
        The matrix A0 in the QCQP.
    s0 : np.ndarray
        The vector s in the QCQP.
    c0 : float
        The constant c in the QCQP.
    A1 : np.ndarray
        The matrix A1 in the QCQP.
    A2 : np.ndarray or sp.csc_array, default is sparse identity
        The matrix A2 in the QCQP.
    s1 : np.ndarray
        The vector s1 in the QCQP.
    Pdiags : np.ndarray
        The diagonal elements of the projection matrices P_j, as columns of a matrix Pdiags
        The second column j should always be one such that A0 + lambda A1 P_j is positive semidefinite for sufficiently large constant lambda.
    verbose : float
        The verbosity level for debugging and logging.
    Acho : sksparse.cholmod.Factor
        The Cholesky factorization of the total A matrix, which is updated when needed.
    current_dual : float
        The current dual solution, which is only updated when the dual problem is solved.
    current_lags : np.ndarray
        The current Lagrangian multipliers, which is only updated when the dual problem is solved.
    current_grad : np.ndarray
        The current grad_lambda of the dual solution, which is only updated when the dual problem is solved.
    current_hess : np.ndarray
        The current hess_lambda of the dual solution, which is only updated when the dual problem is solved.
    """

    def __init__(
        self,
        A0: np.ndarray,
        s0: np.ndarray,
        c0: float,
        A1: np.ndarray,
        s1: np.ndarray,
        Pdiags: np.ndarray,
        A2: np.ndarray = None,
        verbose: float = 0,
    ):

        if A2 is None:
            A2 = sp.eye_array(len(s0), format="csc")

        super().__init__(A0, s0, c0, A1, A2, s1, Pdiags, verbose)

    def __repr__(self):
        return f"DenseSharedProjQCQP of size {self.A0.shape[0]}^2 with {self.Pdiags.shape[1]} projectors."

    def _update_Acho(self, A):
        """
        Updates the Cholesky factorization to be that of the input matrix A.
        Needs to be implemented by subclasses.

        Parameters
        ----------
        A : np.ndarray
            New total A.
        """
        self.Acho = la.cho_factor(A)

    def _Acho_solve(self, b):
        """
        Computes A^{-1} b using Acho.

        Parameters
        ----------
        b : np.ndarray
            Right-hand-side of linear system.

        Returns
        -------
        A^{-1} b
        """
        return la.cho_solve(self.Acho, b)

    def is_dual_feasible(self, lags: np.ndarray) -> bool:
        """
        Checks if a set of Lagrange multipliers is dual feasible by attempting a Cholesky decomposition.

        Arguments
        ---------
        lags: np.ndarray
            A 1-dimensional numpy array of Lagrange multipliers.

        Returns
        -------
        bool
            True if the total A matrix is positive semidefinite (dual feasible).
        """
        A = self._get_total_A(lags)
        try:
            la.cho_factor(A)
            return True
        except la.LinAlgError:
            return False


"""
Generalized Constraint Descent (GCD) methods below.

GCD is a method for tightening dual bounds on a shared projection QCQP
based on iteratively
1. adding of new shared projection constraints that tighten the bound
2. merging of old constraints that does not change the bound to keep total constraint number small.
For usage examples, see the notebooks examples/LDOS_gcd and examples/verlan/LDOS_verlan
For more mathematical details, see Appendix B of https://arxiv.org/abs/2504.10469
"""


def merge_lead_constraints(QCQP: _SharedProjQCQP, merged_num: int = 2) -> None:
    """
    Merge the first m constraints of QCQP into a single constraint.
    Also adjust the Lagrange multipliers so the dual value is the same.

    Parameters
    ----------
    QCQP : _SharedProjQCQP
        QCQP for which we merge the leading constraints.
    merged_num : int (optional, default 2)
        Number of leading constraints that we are merging together; should be at least 2.

    Raises
    ------
    ValueError
        If merged_num < 2 or if there are insufficient constraints for merging.
    """
    x_size, cstrt_num = QCQP.Pdiags.shape
    if merged_num < 2:
        raise ValueError("Need at least 2 constraints for merging.")
    if cstrt_num < merged_num:
        raise ValueError("Number of constraints insufficient for size of merge.")

    new_Pdiags = np.zeros((x_size, cstrt_num - merged_num + 1), dtype=complex)
    new_lags = np.zeros(cstrt_num - merged_num + 1, dtype=float)
    new_Pdiags[:, 0] = QCQP.Pdiags[:, :merged_num] @ QCQP.current_lags[:merged_num]

    # normalize merged Pdiag
    Pnorm = la.norm(new_Pdiags[:, 0])
    new_Pdiags[:, 0] /= Pnorm
    new_lags[0] = Pnorm

    # put other constraints in
    new_Pdiags[:, 1:] = QCQP.Pdiags[:, merged_num:]
    new_lags[1:] = QCQP.current_lags[merged_num:]

    # update QCQP
    if hasattr(QCQP, "precomputed_As"):
        # updated precomputed_As
        QCQP.precomputed_As[merged_num - 1] *= QCQP.current_lags[merged_num - 1]
        for i in range(merged_num - 1):
            QCQP.precomputed_As[merged_num - 1] += (
                QCQP.precomputed_As[i] * QCQP.current_lags[i]
            )
        QCQP.precomputed_As[merged_num - 1] /= Pnorm
        del QCQP.precomputed_As[: merged_num - 1]

    if hasattr(QCQP, "Fs"):
        new_Fs = np.zeros((x_size, cstrt_num - merged_num + 1), dtype=complex)
        new_Fs[:, 0] = QCQP.A2.conj().T @ (new_Pdiags[:, 0].conj() * QCQP.s1)
        new_Fs[:, 1:] = QCQP.Fs[:, merged_num:]
        QCQP.Fs = new_Fs

    QCQP.Pdiags = new_Pdiags
    QCQP.current_lags = new_lags
    QCQP.current_grad = QCQP.current_hess = (
        None  # in principle can merge dual derivatives but leave it undone for now
    )


def add_constraints(
    QCQP: _SharedProjQCQP, added_Pdiag_list: list, orthonormalize: bool = True
) -> None:
    """
    Method that adds new shared projection constraints into an existing QCQP.

    Parameters
    ----------
    QCQP : _SharedProjQCQP
        QCQP for which the new constraints are added in.
    added_Pdiag_list : list
        List of 1d numpy arrays that are the new constraints to be added in.
    orthonormalize : bool, optional
        If true, assume that QCQP has orthonormal constraints and keeps it that way.
    """
    x_size, cstrt_num = QCQP.Pdiags.shape
    added_Pdiag_num = len(added_Pdiag_list)

    if QCQP.current_lags is not None:
        new_lags = np.zeros(cstrt_num + added_Pdiag_num, dtype=float)
        new_lags[:cstrt_num] = QCQP.current_lags
    else:
        new_lags = None

    new_Pdiags = np.zeros((x_size, cstrt_num + added_Pdiag_num), dtype=complex)
    new_Pdiags[:, :cstrt_num] = QCQP.Pdiags
    if not orthonormalize:
        for m, added_Pdiag in enumerate(added_Pdiag_list):
            new_Pdiags[:, cstrt_num + m] = added_Pdiag

    else:
        for m, added_Pdiag in enumerate(added_Pdiag_list):
            # do Gram-Schmidt orthogonalization for each added Pdiag
            for j in range(cstrt_num + m):
                added_Pdiag -= CRdot(new_Pdiags[:, j], added_Pdiag) * new_Pdiags[:, j]
            added_Pdiag /= la.norm(added_Pdiag)

            new_Pdiags[:, cstrt_num + m] = added_Pdiag

    # update QCQP
    if hasattr(QCQP, "precomputed_As"):
        # updated precomputed_As
        for added_Pdiag in added_Pdiag_list:
            QCQP.precomputed_As.append(
                Sym(QCQP.A1 @ sp.diags_array(added_Pdiag, format="csr") @ QCQP.A2)
            )

    if hasattr(QCQP, "Fs"):
        new_Fs = np.zeros((x_size, cstrt_num + added_Pdiag_num), dtype=complex)
        new_Fs[:, :cstrt_num] = QCQP.Fs
        new_Fs[:, cstrt_num:] = (
            QCQP.A2.conj().T @ (new_Pdiags[:, cstrt_num:].conj().T * QCQP.s1).T
        )
        QCQP.Fs = new_Fs

    QCQP.Pdiags = new_Pdiags
    QCQP.current_lags = new_lags
    QCQP.current_grad = QCQP.current_hess = None


def run_gcd(
    QCQP: _SharedProjQCQP,
    max_cstrt_num: int = 10,
    orthonormalize: bool = True,
    opt_params=None,
    max_gcd_iter_num=50,
    gcd_iter_period=5,
    gcd_tol=1e-2,
):
    """
    Perform generalized constraint descent to gradually refine dual bound on QCQP.

    At each GCD iteration, add two new constraints:
    1.a constraint generated so the corresponding dual derivative is large,
    to hopefully tighten the dual bound
    2. a constraint generated so the corresponding derivative of the smallest
    Lagrangian quadratic form eigenvalue is large, to help the dual optimization
    navigate the semi-definite boundary

    If the total number of constraints is larger than max_gcd_cstrt_num combine
    the earlier constraints to keep the total number of constraints fixed. Setting
    max_cstrt_num large enough will eventually result in evaluating the dual bound
    with all possible constraints, which gives the tightest bound but may be extremely
    expensive. The goal of GCD is to approximate this tightest bound with greatly reduced
    computational cost.

    Parameters
    ----------
    QCQP : _SharedProjQCQP
        The SharedProjQCQP for which we compute and refine dual bounds.
    max_cstrt_num : int, optional
        The maximum constraint number for QCQP. The default is 10.
    orthonormalize : bool, optional
        Whether or not to orthonormalize the constraint projectors. The default is True.
    opt_params : dict, optional
        The opt_params for the internal _Optimizer run at every GCD iteration.
    max_gcd_iter_num : int, optional
        Maximum number of GCD iterations, by default 50.
    gcd_iter_period : int, optional
        Period for checking convergence, by default 5.
    gcd_tol : float, optional
        Tolerance for convergence, by default 1e-2.

    Notes
    -----
    TODO: formalize optimization and convergence parameters.
    """
    # since GCD is constantly changing the constraints, no need for many fake source iterations
    OPT_PARAMS_DEFAULTS = {"max_restart": 1}
    if opt_params is None:
        opt_params = {}
    opt_params = {**OPT_PARAMS_DEFAULTS, **opt_params}

    # get to feasible point
    QCQP.current_lags = QCQP.find_feasible_lags()
    if orthonormalize:
        # orthonormalize QCQP
        # informally checked for correctness
        x_size, cstrt_num = QCQP.Pdiags.shape
        realext_Pdiags = np.zeros((2 * x_size, cstrt_num), dtype=float)
        realext_Pdiags[:x_size, :] = np.real(QCQP.Pdiags)
        realext_Pdiags[x_size:, :] = np.imag(QCQP.Pdiags)
        realext_Pdiags_Q, realext_Pdiags_R = la.qr(realext_Pdiags, mode="economic")
        QCQP.Pdiags = realext_Pdiags_Q[:x_size, :] + 1j * realext_Pdiags_Q[x_size:, :]
        QCQP.current_lags = realext_Pdiags_R @ QCQP.current_lags
        QCQP.compute_precomputed_values()

    ## gcd loop
    gcd_iter_num = 0
    gcd_prev_dual = np.inf
    while True:
        gcd_iter_num += 1
        # solve current dual problem
        QCQP.solve_current_dual_problem(
            "newton", init_lags=QCQP.current_lags, opt_params=opt_params
        )
        print(
            f"At GCD iteration #{gcd_iter_num}, best dual bound found is {QCQP.current_dual}."
        )

        ## termination conditions
        if gcd_iter_num > max_gcd_iter_num:
            break
        if gcd_iter_num % gcd_iter_period == 0:
            if gcd_prev_dual - QCQP.current_dual < gcd_tol * abs(gcd_prev_dual):
                break
            gcd_prev_dual = QCQP.current_dual

        ## generate max dualgrad constraint
        maxViol_Pdiag = (2 * QCQP.s1 - (QCQP.A1.conj().T @ QCQP.current_xstar)) * (
            QCQP.A2 @ QCQP.current_xstar
        ).conj()

        ## generate min A eig constraint
        minAeigv, minAeigw = QCQP._get_PSD_penalty(QCQP.current_lags)
        minAeig_Pdiag = (QCQP.A1.conj().T @ minAeigv) * (QCQP.A2 @ minAeigv).conj()
        # use the same relative weights for minAeig_Pdiag as maxViol_Pdiag
        minAeig_Pdiag /= np.sqrt(np.real(minAeig_Pdiag.conj() * minAeig_Pdiag))
        minAeig_Pdiag * np.sqrt(np.real(maxViol_Pdiag.conj() * maxViol_Pdiag))
        # informally checked that minAeigw increases when increasing multiplier of minAeig_Pdiag

        ## add new constraints
        QCQP.add_constraints([maxViol_Pdiag, minAeig_Pdiag])
        # informally checked that new constraints are added in orthonormal fashion

        ## merge old constraints if necessary
        if QCQP.Pdiags.shape[1] > max_cstrt_num:
            QCQP.merge_lead_constraints(
                merged_num=QCQP.Pdiags.shape[1] - max_cstrt_num + 1
            )
