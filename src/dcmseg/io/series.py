from dcmseg.pacs import OrthancClient
import highdicom as hd
import pydicom as pd

def read_dcms(series_id: str) -> list[pd.FileDataset]:
    orthanc = OrthancClient()
    instances = orthanc.series(series_id)['Instances']

    dcms_raw = [orthanc.instance_file(instance) for instance in instances]
    
    dcms = [pd.dcmread(dcm) for dcm in dcms_raw]
        
    return dcms

def make_series(series_id: str):
    images : list[pd.FileDataset] = read_dcms(series_id)

    vol = hd.get_volume_from_series(images)

    print(vol.spatial_shape)
