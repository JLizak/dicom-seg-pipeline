import pydicom

data_path = "../data/duke_breast_cancer_mri/Breast_MRI_001/97538/41458/002c0b46-b9c9-4950-b92c-6027345a4110.dcm"

ds = pydicom.dcmread(data_path)

patient_name = str(ds.PatientName)

pixel_spacing = ds.PixelSpacing

print(pixel_spacing)
