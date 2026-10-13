from dcmseg.pacs import OrthancClient
import highdicom as hd
import pydicom as pd
from io import BytesIO
import argparse

def read_dcms(series_id: str) -> list[pd.FileDataset]:
    orthanc = OrthancClient()
    instances = orthanc.series(series_id)['Instances']

    dcms_raw = [orthanc.instance_file(instance) for instance in instances]
    
    dcms = [pd.dcmread(BytesIO(dcm)) for dcm in dcms_raw]
        
    return dcms

def make_series(series_id: str):
    images : list[pd.FileDataset] = read_dcms(series_id)

    vol = hd.get_volume_from_series(images)

    print(vol.spatial_shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('series_id')

    args = parser.parse_args()
    series_id = args.series_id

    make_series(series_id)
