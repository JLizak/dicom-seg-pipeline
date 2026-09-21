"""Przejdź po hierarchii Patient → Study → Series → Instance w Orthancu i wypisz pierwszą instancję."""

import json

from dcmseg.pacs import OrthancClient


def explore():
    orthanc = OrthancClient()

    first_patient_uid = orthanc.patients()[0]
    patient_0 = orthanc.patient(first_patient_uid)

    patient_0_study_0 = patient_0["Studies"][0]
    study = orthanc.study(patient_0_study_0)

    series_0 = study["Series"][0]
    series = orthanc.series(series_0)

    instance_0_uid = series["Instances"][0]
    print(json.dumps(orthanc.instance(instance_0_uid), indent=2))


if __name__ == "__main__":
    explore()
