import nibabel as nib


class NiftiImage(object):

    def __init__(self, nii_file_path):
        self.img = nib.load(nii_file_path)  # type: nib.nifti1.Nifti1Image

    def convert_to_numpy_array(self):
        return self.img.get_data()

