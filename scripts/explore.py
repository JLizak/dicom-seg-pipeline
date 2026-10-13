from dcmseg.pacs import OrthancClient

def explore():
    orthanc = OrthancClient()
    patients = orthanc.patients()
    
    patient1 = orthanc.patient(patients[0])
    
    study1 = orthanc.study(patient1['Studies'][0])
    
    series1 = orthanc.series(study1["Series"][0])

    instance1 = orthanc.instance(series1["Instances"][0]) 
    
    print(study1)
if __name__ == '__main__':
    explore()
