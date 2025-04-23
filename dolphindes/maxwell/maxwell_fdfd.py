"""
Maxwell solver for E-fields in 2D. Main purpose is to calculate source fields and vacuum Green's functions for bounds calculations.
Inspired by the Ceviche and EMopt solvers.

Warning: this is not meant to be a high-performance Maxwell solver! Most bounds calculations require a single solve 
for the initial source (if not known) and one for the Green's function. 
"""

__all__ = ['TM_FDFD']

import numpy as np 
import scipy.sparse as sp 

class Maxwell_FDFD():
    """
    Attributes
    ----------
    omega : complex
        circular frequency, can be complex to allow for finite bandwidth effects
    wavelength : float
        wavelength of the wave, in units of 1
    dx : float
        finite difference grid pixel size in x-axis, in units of 1 
    dy : float
        finite difference grid pixel size in y-axis, in units of 1
    Nx : int
        number of pixels along the x direction.
    Ny : int
        number of pixels along the y direction.
    Npmlx : int or tuple
        number of pixels in the PML region, x direction
    Npmly : int or tuple
        number of pixels in the PML region, y direction
    bloch_x : float, optional
        x-direction phase shift associated with the periodic boundary condtions. The default is 0.0.
    bloch_y : float, optional
        y-direction phase shift associated with the periodic boundary condtions. The default is 0.0.
        
    """
    def __init__(self, omega, Nx, Ny, Npmlx, Npmly, dx, dy, bloch_x=0.0, bloch_y=0.0):
        self.omega = omega 

        if not isinstance(Nx, int) or not isinstance(Ny, int) or not isinstance(Npmlx, int) or not isinstance(Npmly, int):
            import warnings
            warnings.warn("Nx, Ny, Npmlx, and Npmly should be integers. Automatically converting them to integers.")
            Nx, Ny, Npmlx, Npmly = int(Nx), int(Ny), int(Npmlx), int(Npmly)
        
        self.Nx = Nx
        self.Ny = Ny
        self.Npmlx = Npmlx
        self.Npmly = Npmly
        self.dx = dx
        self.dy = dy
        self.EPSILON_0 = 1.0
        self.MU_0 = 1.0
        self.C_0 = 1.0
        self.ETA_0 = 1.0
        self.bloch_x = bloch_x
        self.bloch_y = bloch_y
        self.k = self.omega / self.C_0

        self.nonpmlNx = self.Nx - 2 * self.Npmlx 
        self.nonpmlNy = self.Ny - 2 * self.Npmly

        assert (self.nonpmlNx > 0) and (self.nonpmlNy > 0), "Non-PML grid size must be positive. Check Nx, Ny, Npmlx, and Npmly values."

class TM_FDFD(Maxwell_FDFD):
    """
    Finite-difference frequency-domain solver for TM fields in 2D, with PMLs on the boundaries.
    
    Attributes
    ----------
    All of the attributes of the parent class Maxwell_FDFD, plus:
    dl : float
        finite difference grid pixel size, in units of 1
    M0 : sp.csc_array
        Maxwell operator in sparse matrix format, representing the operator ∇x∇x - omega^2 I in 2D for TM fields.
    """

    def __init__(self, omega, Nx, Ny, Npmlx, Npmly, dl, bloch_x=0.0, bloch_y=0.0):
        dx, dy = dl, dl 
        super().__init__(omega, Nx, Ny, Npmlx, Npmly, dx, dy, bloch_x, bloch_y)
        self.dl = dl 
        self.M0 = self._make_TM_Maxwell_Operator(self.Nx, self.Ny, self.Npmlx, self.Npmly)

    def _make_TM_Maxwell_Operator(self, Nx, Ny, Npmlx, Npmly) -> sp.csc_array:
        """ Assembles the Maxwell operator ∇x∇x - omega^2 I in 2D for TM fields, with PMLs on the boundaries 
        
        Returns
        -------
        M : sp.csc_array
            Maxwell operator in sparse array format.
        
        """
        def make_Dxf(dL: float, shape: tuple, bloch_x: complex = 0.0) -> sp.csc_array:
            """ Forward derivative in x """
            Nx, Ny = shape
            phasor_x = np.exp(1j * bloch_x)
            Dxf = sp.diags_array([-1, 1], offsets=[0, 1], shape=(Nx, Nx), dtype=complex) + sp.diags_array([phasor_x], offsets=[-Nx+1], shape=(Nx, Nx), dtype=complex)
            Dxf = 1 / dL * sp.kron(Dxf, sp.eye(Ny), format="csc")
            return Dxf

        def make_Dxb(dL: float, shape: tuple, bloch_x: complex = 0.0) -> sp.csc_array:
            """ Backward derivative in x """
            Nx, Ny = shape
            phasor_x = np.exp(1j * bloch_x)
            Dxb = sp.diags_array([1, -1], offsets=[0, -1], shape=(Nx, Nx), dtype=complex) + sp.diags_array([-np.conj(phasor_x)], offsets=[Nx-1], shape=(Nx, Nx), dtype=complex)
            Dxb = 1 / dL * sp.kron(Dxb, sp.eye(Ny), format="csc")
            return Dxb

        def make_Dyf(dL: float, shape: tuple, bloch_y: complex = 0.0) -> sp.csc_array:
            """ Forward derivative in y """
            Nx, Ny = shape
            phasor_y = np.exp(1j * bloch_y)
            Dyf = sp.diags_array([-1, 1], offsets=[0, 1], shape=(Ny, Ny), dtype=complex) + sp.diags_array([phasor_y], offsets=[-Ny+1], shape=(Ny, Ny), dtype=complex)
            Dyf = 1 / dL * sp.kron(sp.eye(Nx), Dyf, format="csc")
            return Dyf

        def make_Dyb(dL: float, shape: tuple, bloch_y: complex = 0.0) -> sp.csc_array:
            """ Backward derivative in y """
            Nx, Ny = shape
            phasor_y = np.exp(1j * bloch_y)
            Dyb = sp.diags_array([1, -1], offsets=[0, -1], shape=(Ny, Ny), dtype=complex) + sp.diags_array([-np.conj(phasor_y)], offsets=[Ny-1], shape=(Ny, Ny), dtype=complex)
            Dyb = 1 / dL * sp.kron(sp.eye(Nx), Dyb, format="csc")
            return Dyb
    
        def sig_w(l: float, dw: float, m: float = 3, lnR: float = -30) -> float:
            """ Fictional conductivity for adding PML, note that these values might need tuning """
            sig_max = -(m + 1) * lnR / (2 * self.ETA_0 * dw)
            return sig_max * (l / dw)**m

        def s_value(l: float, dw: float, omega: complex) -> complex:
            """ S-value to use in the S-matrices """
            return 1 + 1j * sig_w(l, dw) / (omega * self.EPSILON_0)
        
        def create_sfactor_f(omega: complex, dL: float, N: int, N_pml: int, dw: float) -> np.ndarray:
            """ S-factor profile for forward derivative matrix """
            sfactor_array = np.ones(N, dtype=complex)
            for i in range(N):
                if i <= N_pml:
                    sfactor_array[i] = s_value(dL * (N_pml - i + 0.5), dw, omega)
                elif i > N - N_pml:
                    sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 0.5), dw, omega)
            return sfactor_array

        def create_sfactor_b(omega: complex, dL: float, N: int, N_pml: int, dw: float) -> np.ndarray:
            """ S-factor profile for backward derivative matrix """
            sfactor_array = np.ones(N, dtype=complex)
            for i in range(N):
                if i <= N_pml:
                    sfactor_array[i] = s_value(dL * (N_pml - i + 1), dw, omega)
                elif i > N - N_pml:
                    sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 1), dw, omega)
            return sfactor_array
    
        def create_sfactor(dir: str, omega: complex, dL: float, N: int, N_pml: int) -> np.ndarray:
            """ creates the S-factor cross section needed in the S-matrices """

            #  for no PNL, this should just be zero
            if N_pml == 0:
                return np.ones(N, dtype=complex)

            # otherwise, get different profiles for forward and reverse derivative matrices
            dw = N_pml * dL
            if dir == 'f':
                return create_sfactor_f(omega, dL, N, N_pml, dw)
            elif dir == 'b':
                return create_sfactor_b(omega, dL, N, N_pml, dw)
            else:
                raise ValueError(f"Dir value {dir} not recognized")
            
        def create_S_matrices(omega: complex, shape: tuple[int, int], npml: tuple[int, int], dL: float) -> tuple[sp.csc_array, sp.csc_array, sp.csc_array, sp.csc_array]:
            """ Makes the 'S-matrices'.  When dotted with derivative matrices, they add PML """

            # strip out some information needed
            Nx, Ny = shape
            N = Nx * Ny
            Nx_pml, Ny_pml = npml    

            # Create the sfactor in each direction and for 'f' and 'b'
            s_vector_x_f = create_sfactor('f', omega, dL, Nx, Nx_pml)
            s_vector_x_b = create_sfactor('b', omega, dL, Nx, Nx_pml)
            s_vector_y_f = create_sfactor('f', omega, dL, Ny, Ny_pml)
            s_vector_y_b = create_sfactor('b', omega, dL, Ny, Ny_pml)

            # Fill the 2D space with layers of appropriate s-factors
            Sx_f_2D = np.zeros(shape, dtype=complex)
            Sx_b_2D = np.zeros(shape, dtype=complex)
            Sy_f_2D = np.zeros(shape, dtype=complex)
            Sy_b_2D = np.zeros(shape, dtype=complex)

            # insert the cross sections into the S-grids (could be done more elegantly)
            for i in range(0, Ny):
                Sx_f_2D[:, i] = 1 / s_vector_x_f
                Sx_b_2D[:, i] = 1 / s_vector_x_b
            for i in range(0, Nx):
                Sy_f_2D[i, :] = 1 / s_vector_y_f
                Sy_b_2D[i, :] = 1 / s_vector_y_b

            # Reshape the 2D s-factors into a 1D s-vector
            Sx_f_vec = Sx_f_2D.flatten()
            Sx_b_vec = Sx_b_2D.flatten()
            Sy_f_vec = Sy_f_2D.flatten()
            Sy_b_vec = Sy_b_2D.flatten()

            # Construct the 1D total s-vector into a diagonal matrix using diags_array instead of spdiags
            Sx_f = sp.dia_array((Sx_f_vec, 0), shape=(N, N))
            Sx_b = sp.dia_array((Sx_b_vec, 0), shape=(N, N))
            Sy_f = sp.dia_array((Sy_f_vec, 0), shape=(N, N))
            Sy_b = sp.dia_array((Sy_b_vec, 0), shape=(N, N))
            
            return Sx_f, Sx_b, Sy_f, Sy_b

        shape = (Nx, Ny)

        Dxf = make_Dxf(self.dl, shape, bloch_x=self.bloch_x)
        Dxb = make_Dxb(self.dl, shape, bloch_x=self.bloch_x)
        Dyf = make_Dyf(self.dl, shape, bloch_y=self.bloch_y)
        Dyb = make_Dyb(self.dl, shape, bloch_y=self.bloch_y)

        Sxf, Sxb, Syf, Syb = create_S_matrices(self.omega, shape, (Npmlx, Npmly), self.dl)
        
        #dress the derivative functions with pml
        Dxf = Sxf @ Dxf
        Dxb = Sxb @ Dxb

        Dyf = Syf @ Dyf
        Dyb = Syb @ Dyb

        M = sp.csc_array(-Dxf @ Dxb - Dyf @ Dyb - self.EPSILON_0*self.omega**2 * sp.eye(Nx*Ny))
        return M
    
    def _get_diagM_from_chigrid(self, chigrid: np.ndarray) -> sp.dia_array:
        """ get the diagonal part of the Maxwell operator from the material susceptibility chigrid (flattens it) """
        return -sp.diags_array(chigrid.flatten() * self.omega**2, format='dia')
    
    def get_TM_dipole_field(self, cx: int, cy: int, chigrid: np.ndarray = None) -> np.ndarray:
        """
        Get the field of a TM dipole source at position (cx, cy) with material distribution given by chigrid.

        Parameters
        ----------
        cx : int
            x-coordinate of the dipole source.
        cy : int
            y-coordinate of the dipole source.
        chigrid : np.ndarray (dtype complex), optional
            spatial distribution of material susceptibility. The default is None, corresponding to vacuum.

        Returns
        -------
        Ez : np.ndarray (dtype complex)
            Field of the dipole source at position (cx, cy).
        """
        sourcegrid = np.zeros((self.Nx, self.Ny), dtype=complex)
        sourcegrid[cx, cy] = 1.0 / (self.dl**2)
        return self.get_TM_field(sourcegrid, chigrid)

    def get_TM_field(self, sourcegrid: np.ndarray, chigrid: np.ndarray = None) -> np.ndarray:
        """
        Get the field of a TM source at positions in sourcegrid with material distribution given by chigrid.
        
        Parameters
        ----------
        sourcegrid : np.ndarray (dtype complex)
            spatial distribution of the source. 
        chigrid : np.ndarray (dtype complex), optional
            spatial distribution of material susceptibility. The default is None, corresponding to vacuum.

        Returns
        -------
        Ez : np.ndarray (dtype complex)
            Field of the dipole source.

        """
        M = self.M0 + self._get_diagM_from_chigrid(chigrid) if chigrid is not None else self.M0
        RHS = 1j * self.omega * sourcegrid.flatten()
        Ez = np.reshape(sp.linalg.spsolve(M, RHS), (self.Nx, self.Ny))
        return Ez

    def get_TM_G_ba(self, A_mask: np.ndarray, B_mask: np.ndarray) -> np.ndarray:
        """
        Compute the vacuum Green’s function G_{BA} mapping sources in region A to fields in region B.

        This routine exploits translational symmetry by embedding two copies of the non-PML domain
        into a larger “big” grid. A single dipole solve at the center of the big grid produces
        a field map Ezfield. For each source location in the design (A_mask), we extract the
        corresponding window of size (nonpmlNx × nonpmlNy) and sample at the observation mask B_mask.

        Parameters
        ----------
        A_mask : np.ndarray of bool, shape (Nx, Ny)
            Mask specifying the source/design region in the full grid.
        B_mask : np.ndarray of bool, shape (Nx, Ny)
            Mask specifying the observation region in the full grid.

        Returns
        -------
        G_od : np.ndarray of complex, shape (n_obs, n_src)
            Green’s function matrix where each column is the field at B_mask
            due to a unit dipole at a location in A_mask.
        """
        # validate masks against the full grid
        assert A_mask.shape == (self.Nx, self.Ny)
        assert B_mask.shape == (self.Nx, self.Ny)

        # restrict masks to the non-PML interior
        A_mask_s = A_mask[self.Npmlx:self.Npmlx + self.nonpmlNx,
                          self.Npmly:self.Npmly + self.nonpmlNy]
        B_mask_s = B_mask[self.Npmlx:self.Npmlx + self.nonpmlNx,
                          self.Npmly:self.Npmly + self.nonpmlNy]

        # dimension checks
        assert A_mask_s.shape == (self.nonpmlNx, self.nonpmlNy)
        assert B_mask_s.shape == (self.nonpmlNx, self.nonpmlNy)

        # build a “big” grid that can slide the small domain around
        bigNx = 2 * self.nonpmlNx - 1 + 2 * self.Npmlx
        bigNy = 2 * self.nonpmlNy - 1 + 2 * self.Npmly
        bigcx = self.Npmlx + self.nonpmlNx - 1  # center x index
        bigcy = self.Npmly + self.nonpmlNy - 1  # center y index

        # assemble vacuum Maxwell operator on the big grid
        A = self._make_TM_Maxwell_Operator(bigNx, bigNy,
                                           self.Npmlx, self.Npmly)

        # place a unit dipole source at the center of the big grid
        sourcegrid = np.zeros((bigNx, bigNy), dtype=complex)
        sourcegrid[bigcx, bigcy] = 1.0 / self.dl
        RHS = 1j * self.omega * sourcegrid.flatten()

        # solve once for Ezfield on the big grid
        Ezfield = np.reshape(sp.linalg.spsolve(A, RHS), (bigNx, bigNy))

        # indices of design points in the small grid
        design_idx = np.argwhere(A_mask_s)
        n_src = design_idx.shape[0]
        n_obs = int(np.sum(B_mask_s))
        G_od = np.zeros((n_obs, n_src), dtype=complex)

        # for each design point, extract the corresponding sub-window in Ezfield
        for i, (ix, iy) in enumerate(design_idx):
            ulx = bigcx - ix  # upper-left corner x of the small grid in big grid
            uly = bigcy - iy  # upper-left corner y 
            window = Ezfield[ulx : ulx + self.nonpmlNx,
                             uly : uly + self.nonpmlNy]
            G_od[:, i] = window[B_mask_s]

        # scale to get the true vacuum Green’s function for TM polarization
        G_od *= self.dl * (-1j * self.k / self.ETA_0)
        return G_od


    def get_Gaainv(self, A_mask: np.ndarray, chigrid: np.ndarray = None) -> tuple[np.ndarray, sp.csc_array]:
        """
        Compute the inverse Green’s function on region A, G_{AA}^{-1}, using a Woodbury identity.

        We partition the full Maxwell operator M into blocks corresponding to region A (design)
        and its complement B (background):
            M = [[A, B],
                 [C, D]]
        Then G_{AA}^{-1} = D - C A^{-1} B, up to a multiplicative constant MU_0 / k^2.

        Parameters
        ----------
        A_mask : np.ndarray of bool, shape (Nx, Ny)
            Mask for the design region A.
        chigrid : np.ndarray of complex, optional
            Material susceptibility distribution. If provided, M = M0 + diag(ω² χ).

        Returns
        -------
        G_ddinv : sp.csc_array of shape (n_src, n_src)
            The inverse Green’s function on region A.
        M : sp.csc_array
            The full Maxwell operator used in the computation.
        """
        # assemble full Maxwell operator (with materials if given)
        M = self.M0 if chigrid is None else self.M0 + self._get_diagM_from_chigrid(chigrid)

        # flatten masks and get index lists for design (A) and background (B)
        flatA = A_mask.flatten()
        designInd = np.nonzero(flatA)[0]
        backgroundInd = np.nonzero(~flatA)[0]

        # extract blocks A, B, C, D from M
        A = (M[:, backgroundInd])[backgroundInd, :]
        B = (M[:, designInd])[backgroundInd, :]
        C = (M[:, backgroundInd])[designInd, :]
        D = (M[designInd, :])[:, designInd]

        # solve A * X = B  → X = A^{-1} B
        AinvB = sp.linalg.spsolve(A, B)

        # Woodbury: G_{AA}^{-1} = D - C A^{-1} B
        Gfac = self.MU_0 / self.k**2
        G_ddinv = (D - (C @ AinvB)) * Gfac

        return G_ddinv, M