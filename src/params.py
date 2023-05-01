import astropy.units as u
import astropy.constants as const
import numpy as np


# =========
# Variables
# =========
folder = r'/home/jamiechang917/cloud/Scripts/f2r3d/data/mydisk'
snap_index = 700
star_mass = 1  # solar mass
star_radius = 1.5  # solar radius
star_temp = 5780  # K
planet_r0 = 10
unit_length = 100 * u.au
unit_mass = 1 * u.solMass
dustopac = 'silicate'
gasspec = 'co'
nphot = 1e6
nphot_scat = 1e6
nphot_spec = 1e6
dusttogas = 0.01


unit_time = np.sqrt(unit_length**3 / (const.G * unit_mass)).to('s')
unit_velocity = (unit_length / unit_time).to('cm/s')
unit_density = (unit_mass / unit_length**3).to('g/cm^3')
planet_r0 = (planet_r0 * unit_length).to('cm')
planet_v0 = (np.sqrt(const.G * star_mass * const.M_sun / planet_r0)).to('cm/s')
