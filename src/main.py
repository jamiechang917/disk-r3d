import os
import shutil
import warnings
import numpy as np
import radmc3dPy as r3d
import params as par
import astropy.units as u
import astropy.constants as const

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import FuncFormatter


def read_variables():
    '''
    read variables.par file and return a dict
    folder: folder path (fargo3d output folder)
    '''
    vars = {}
    try:
        with open(par.folder + '/variables.par', 'r') as f:
            # read lines and save as dict
            for line in f.readlines():
                line = line.strip()
                if line:
                    key, value = line.split('\t')
                    vars[key] = value
    except:
        print('Cannot find variables.par file in folder: ' + par.folder)
    return vars


def init_setup():
    vars = read_variables()
    gasabun = 0
    if par.gasspec == 'co':
        gasabun = 1e-4
    elif par.gasspec == 'hco+':
        gasabun = 1e-8
    else:
        assert False, 'gasspec not supported.'
    xbound = (float(vars['YMIN'])*par.unit_length.to('cm').value,
              float(vars['YMAX'])*par.unit_length.to('cm').value)
    model = r3d.setup.radmc3dModel(
        model='ppdisk',
        mstar=f'[{par.star_radius}*ms]',
        rstar=f'[{par.star_radius}*rs]',
        tstar=f'[{par.star_temp}]',
        crd_sys="\'sph\'",
        grid_type='0',  # 0: regular grid, 1: octree AMR
        # number of grid points in x direction (radial in spherical coordinates)
        nx=f'{vars["NY"]}',
        # number of grid points in y direction (colatitude in spherical coordinates)
        ny=f'{vars["NZ"]}',
        # number of grid points in z direction (azimuth in spherical coordinates)
        nz=f'{vars["NX"]}',
        # boundaries in x direction (radial in spherical coordinates)
        xbound=f'[{xbound[0]}, {xbound[1]}]',
        # boundaries in y direction (colatitude in spherical coordinates)
        ybound=f'[{vars["ZMIN"]}, {vars["ZMAX"]}]',
        # boundaries in z direction (azimuth in spherical coordinates)
        zbound=f'[{float(vars["XMIN"])+np.pi}, {float(vars["XMAX"])+np.pi}]',
        dustkappa_ext=f'[\'{par.dustopac}\']',  # dust opacity file
        gasspec_mol_abun=f'[{gasabun}]',  # gas abundance
        gasspec_mol_name=f'[\'{par.gasspec}\']',  # gas name
        gasspec_mol_dbase_type='[\'leiden\']',
        gasspec_vturb='0.2e5',  # turbulence velocity
        nphot=f'{int(par.nphot)}',  # number of photons
        nphot_scat=f'{int(par.nphot_scat)}',
        nphot_spec=f'{int(par.nphot_spec)}',
        # 0 - no scattering, 1 - isotropic scattering, 2 - anizotropic scattering
        scattering_mode_max='0',
        tgas_eq_tdust='1',  # Take the dust temperature to identical to the gas temperature
        dusttogas=f'{par.dusttogas}',  # dust to gas ratio
        xres_nlev='1',  # leave to 1
        xres_nspan='1',  # leave to 1
        xres_nstep='1',  # leave to 1
        lines_mode='1'

    )
    model.writeRadmc3dInp()  # generate radmc3d.inp file

    # download LAMDA molecular file
    if not os.path.exists(f"molecule_{par.gasspec}.inp"):
        try:
            os.system(
                f'curl -o molecule_{par.gasspec}.inp https://home.strw.leidenuniv.nl/~moldata/datafiles/{par.gasspec}.dat')
        except:
            print("Cannot download molecular file!")

    # write lines.inp
    with open("lines.inp", "w") as f:
        f.write("2\n")
        f.write("1\n")
        f.write(f"{par.gasspec}   leiden   0   0   0")

    return


def create_grid():
    '''
    create grid
    '''
    # read params
    modpar = r3d.analyze.readParams()
    ppar = modpar.ppar

    # create grid
    grid = r3d.analyze.radmc3dGrid()
    grid.makeWavelengthGrid(ppar=ppar)
    grid.makeSpatialGrid(
        ppar=ppar)

    # write grid
    grid.writeWavelengthGrid(old=False)
    grid.writeSpatialGrid(old=False)

    return grid


def create_rad_src(grid):
    '''
    create radiation sources
    '''
    # read params
    modpar = r3d.analyze.readParams()
    ppar = modpar.ppar

    # create radiation sources
    radsrc = r3d.analyze.radmc3dRadSources(ppar=ppar, grid=grid)
    radsrc.getStarSpectrum(tstar=ppar['tstar'], rstar=ppar['rstar'])

    # write radiation sources
    radsrc.writeStarsinp(old=False, ppar=ppar)

    return radsrc


def create_field():
    '''
    read fargo3d output
    '''

    # read params
    vars = read_variables()
    modpar = r3d.analyze.readParams()
    ppar = modpar.ppar

    # get planet velocity (vphi)
    planet_info = read_planet_info(snap_index=par.snap_index)
    planet_x, planet_y = planet_info[1] * \
        par.unit_length, planet_info[2] * par.unit_length
    planet_r = np.sqrt(planet_x**2 + planet_y**2)
    planet_vkep = np.sqrt(
        const.G * par.star_mass * const.M_sun / planet_r).to('cm/s')

    # read gasdens file
    gasdens = np.fromfile(
        par.folder + f'/gasdens{par.snap_index}.dat', dtype=np.float64).reshape(int(vars['NZ']), int(vars['NY']), int(vars['NX'])) * \
        par.unit_density.to('g/cm^3').value
    # swap x and y axis because of different definition of (nx, ny, nz) in radmc3d and fargo3d
    gasdens = np.swapaxes(gasdens, 0, 1)
    gasdens = np.expand_dims(gasdens, axis=3)
    # read gasvel file (vx: vphi, vy: vr, vz: vtheta)
    gasvx = np.swapaxes(np.fromfile(
        par.folder + f'/gasvx{par.snap_index}.dat', dtype=np.float64).reshape(int(vars['NZ']), int(vars['NY']), int(vars['NX'])), 0, 1) * \
        par.unit_velocity.to('cm/s').value + planet_vkep.to('cm/s').value
    gasvy = np.swapaxes(np.fromfile(
        par.folder + f'/gasvy{par.snap_index}.dat', dtype=np.float64).reshape(int(vars['NZ']), int(vars['NY']), int(vars['NX'])), 0, 1) * \
        par.unit_velocity.to('cm/s').value
    gasvz = np.swapaxes(np.fromfile(
        par.folder + f'/gasvz{par.snap_index}.dat', dtype=np.float64).reshape(int(vars['NZ']), int(vars['NY']), int(vars['NX'])), 0, 1) * \
        par.unit_velocity.to('cm/s').value
    gasvx = np.expand_dims(gasvx, axis=3)
    gasvy = np.expand_dims(gasvy, axis=3)
    gasvz = np.expand_dims(gasvz, axis=3)
    gasvel = np.concatenate((gasvy, gasvz, gasvx),
                            axis=3)  # (vr, vtheta, vphi)

    data = r3d.analyze.radmc3dData(grid)
    data.rhodust = gasdens * ppar['dusttogas']
    data.rhogas = gasdens
    data.gasvel = gasvel

    # write to radmc3d inputs
    data.writeDustDens(old=False, binary=False)
    # data.writeGasDens(ispec='co', binary=False) # need to be fixed
    data.rhodust /= ppar['dusttogas']

    # calculate number density of CO (assume abundance of 1e-4)
    if par.gasspec == 'co':
        factco = 1e-4 / (2.3*1.67262192e-27)
        data.rhodust *= factco
        _name = f'numberdens_{par.gasspec}.inp'
    else:
        assert False, f"Gas specie {par.gasspec} is not supported yet!"
    data.writeDustDens(fname=_name, old=False, binary=False)
    data.writeGasVel(binary=False)

    return


def create_opac():
    ppar = r3d.analyze.readParams().ppar
    opac = r3d.analyze.radmc3dDustOpac()
    opac.writeMasterOpac(
        ext=ppar['dustkappa_ext'], scattering_mode_max=ppar['scattering_mode_max'], old=False)
    shutil.copy('../src/opac/dustkappa_' +
                str(ppar['dustkappa_ext'][0])+'.inp', '.')


def read_planet_info(snap_index, planet_index=0):
    '''
    read planet info from planet.dat
    return [snap_index, x, y, z, vx, vy, vz, mass, date, rotation rate]
    '''
    data = []
    with open(f'{par.folder}/planet{planet_index}.dat', 'r') as f:
        for line in f:
            data.append(line.split())
    data = np.array(data, dtype=np.float64)
    return data[snap_index]


def plot_snapshot():
    '''
    visualize snapshot and save a pdf file
    '''
    # read params
    vars = read_variables()
    ppar = r3d.analyze.readParams().ppar

    planet_info = read_planet_info(snap_index=par.snap_index)
    planet_x, planet_y = planet_info[1] * \
        par.unit_length, planet_info[2] * par.unit_length
    planet_r = np.sqrt(planet_x**2 + planet_y**2)
    planet_vkep = np.sqrt(
        const.G * par.star_mass * const.M_sun / planet_r).to('cm/s')
    # read gasdens file
    gasdens = np.fromfile(
        par.folder + f'/gasdens{par.snap_index}.dat', dtype=np.float64).reshape(int(vars['NZ']), int(vars['NY']), int(vars['NX'])) * \
        par.unit_density.to('g/cm^3')
    gasvx = np.fromfile(
        par.folder + f'/gasvx{par.snap_index}.dat', dtype=np.float64).reshape(int(vars['NZ']), int(vars['NY']), int(vars['NX'])) * \
        par.unit_velocity.to('cm/s') + planet_vkep.to('cm/s')
    gasvy = np.fromfile(
        par.folder + f'/gasvy{par.snap_index}.dat', dtype=np.float64).reshape(int(vars['NZ']), int(vars['NY']), int(vars['NX'])) * \
        par.unit_velocity.to('cm/s')
    gasvz = np.fromfile(
        par.folder + f'/gasvz{par.snap_index}.dat', dtype=np.float64).reshape(int(vars['NZ']), int(vars['NY']), int(vars['NX'])) * \
        par.unit_velocity.to('cm/s')

    r = np.linspace(ppar['xbound'][0], ppar['xbound']
                    [1], ppar['nx']) * u.cm
    phi = np.linspace(ppar['zbound'][0], ppar['zbound'][1],
                      ppar['nz']) * u.rad
    R, PHI = np.meshgrid(r, phi)
    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)
    # Z is planned to add in the future

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1,
                        right=0.9, top=0.9, bottom=0.1)
    # plt.style.use('dark_background')

    def fmt(x, pos): return '{:.1f}'.format(x)

    # ignore warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1 = ax[0, 0].pcolormesh(X.to('au').value, Y.to('au').value, np.log10(
            gasdens[int(ppar['ny']/2)].T.value),  cmap='inferno')
        p2 = ax[0, 1].pcolormesh(X.to('au').value, Y.to('au').value, gasvx[int(ppar['ny']/2)].to('km/s').value.T,
                                 cmap='RdBu')
        p3 = ax[1, 0].pcolormesh(X.to('au').value, Y.to('au').value, gasvy[int(ppar['ny']/2)].to('m/s').value.T,
                                 cmap='RdBu')
        p4 = ax[1, 1].pcolormesh(X.to('au').value, Y.to('au').value, gasvz[int(ppar['ny']/2)].to('m/s').value.T,
                                 cmap='RdBu')

        cb1 = plt.colorbar(p1, ax=ax[0, 0], fraction=0.048,
                           pad=0, format=FuncFormatter(fmt))
        cb2 = plt.colorbar(p2, ax=ax[0, 1], fraction=0.048,
                           pad=0, format=FuncFormatter(fmt))
        cb3 = plt.colorbar(p3, ax=ax[1, 0], fraction=0.048,
                           pad=0, format=FuncFormatter(fmt))
        cb4 = plt.colorbar(p4, ax=ax[1, 1], fraction=0.048,
                           pad=0, format=FuncFormatter(fmt))
        cb1.set_label(r'log $\rho_{midplane}$ [g/cm$^3$]')
        cb2.set_label(r'$v_{\phi}$ [km/s]')
        cb3.set_label(r'$v_{r}$ [m/s]')
        cb4.set_label(r'$v_{\theta}$ [m/s]')

        for cb in [cb1, cb2, cb3, cb4]:
            cb.ax.tick_params(labelsize=8)
        # set aspect ratio
        for i in range(2):
            ax[i, 0].set_ylabel('y [AU]')
            for j in range(2):
                ax[i, j].set_aspect('equal')
                ax[1, j].set_xlabel('x [AU]')
                ax[i, j].tick_params(labelsize=8)

    plt.savefig(f'gasdens_{par.snap_index}.pdf')
    return


def plot_r3d_image():
    try:
        im = r3d.image.readImage()  # read image.out
    except:
        print("No image.out found.")
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    r3d.image.plotImage(im, au=True, log=False, cmap=mpl.cm.inferno)
    ax.set_xlabel('x [AU]')
    ax.set_ylabel('y [AU]')
    ax.set_aspect('equal')
    plt.savefig('r3d_output.png')
    return


if __name__ == '__main__':
    # create output folder
    if not os.path.exists("output"):
        os.makedirs("output")
        print("Create output folder.")
    os.chdir("output")
    init_setup()
    grid = create_grid()
    radsrc = create_rad_src(grid)
    create_field()
    create_opac()
    plot_snapshot()
