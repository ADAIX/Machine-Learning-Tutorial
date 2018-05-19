import os
import numpy as np
from astropy.io import fits

from sklearn.preprocessing import LabelEncoder


def get_star_gal_data(datadir='../data'):
    catalog_file = os.path.join(datadir, 'star_gal_catalog.fits')
    # Load the SExtractor catalog
    catalog = fits.getdata(catalog_file, ext=2)
    object_class = fits.getdata(catalog_file, ext=3)['OBJCLASS']
    
    cut = object_class != 0
    catalog = catalog[cut]
    object_class = object_class[cut]

    lenc = LabelEncoder()
    y = lenc.fit_transform(object_class)

    mumax = catalog['MU_MAX']
    mag = catalog['MAG_AUTO']
    mumag = mumax - mag

    X = np.concatenate([mag[:, None], mumag[:, None]], axis=1)

    return X, y
