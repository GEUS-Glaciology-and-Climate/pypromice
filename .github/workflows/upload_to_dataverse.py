import argparse
from time import sleep
from os.path import join, relpath
from os import walk, getcwd
import requests, json
from pyDataverse.api import NativeApi
from pyDataverse.models import Datafile

def parse_arguments():
    """ Parses cmd-line arguments """
    parser = argparse.ArgumentParser()

    # Mandatory arguments
    parser.add_argument("-t", "--token", help="Dataverse token.")
    parser.add_argument("-s","--server", help="Dataverse server.")
    parser.add_argument("-d", "--doi", help="Dataset DOI.")
    parser.add_argument("-r", "--repo", help="GitHub repository.")
    parser.add_argument("-e", "--title", help="Amended title of Dataset.")

    # Optional arguments
    parser.add_argument("-i", "--dir", help="Uploads only a specific dir.")
    parser.add_argument(
        "-v", "--remove", help="Remove (delete) all files before upload.", \
        choices=('True', 'TRUE', 'true', 'False', 'FALSE', 'false'), \
        default='true')
    parser.add_argument(
        "-p", "--publish", help="Publish a new dataset version after upload.", \
        choices=('True', 'TRUE', 'true', 'False', 'FALSE', 'false'), \
        default='false')

    args_ = parser.parse_args()
    return args_


def check_dataset_lock(num):
    """ Gives Dataverse server more time for upload """
    if num <= 1:
        print('Lock found for dataset id ' + \
          str(dataset_dbid) + '\nTry again later!')
        return

    query_str = dataverse_server + \
         '/api/datasets/' + str(dataset_dbid) + '/locks/'
    resp_ = requests.get(query_str, auth = (token, ""))
    locks = resp_.json()['data']

    if bool(locks):
        print('Lock found for dataset id ' + \
           str(dataset_dbid) + '\n... sleeping...')
        sleep(2)
        check_dataset_lock(num-1)
    return


if __name__ == '__main__':

    args = parse_arguments()
    token = args.token
    dataverse_server = args.server.strip("/")
    print(f"Using Dataverse server: {dataverse_server}")

    api = NativeApi(dataverse_server , token)
    resp = api.get_dataset(args.doi)
    resp.raise_for_status()
    dataset = resp

    files_list = dataset.json()['data']['latestVersion']['files']
    dataset_dbid = dataset.json()['data']['id']

    if args.remove.lower() == 'true':
        # the following deletes all the files in the dataset
        delete_api = dataverse_server + \
            '/dvn/api/data-deposit/v1.1/swordv2/edit-media/file/'
        for f in files_list:
            fileid = f["dataFile"]["id"]
            resp = requests.delete(
                delete_api + str(fileid), \
                auth = (token  , ""))

    # check if there is a list of dirs to upload
    repo_root = getcwd()
    paths = [repo_root]
    if args.dir:
        dirs = args.dir.strip().replace(",", " ")
        dirs = dirs.split()
        paths = [join(repo_root, d) for d in dirs]

    # the following adds all files from the repository to Dataverse
    for path in paths:
        for root, subdirs, files in walk(path):
            if '.git' in subdirs:
                subdirs.remove('.git')
            if '.github' in subdirs:
                subdirs.remove('.github')
            for f in files:
                df = Datafile()
                df.set({
                    "pid" : args.doi,
                    "filename" : f,
                    "directoryLabel": relpath(root, start='repo'),
                    "description" : \
                      "Uploaded with GitHub Action from {}.".format(
                        args.repo),
                    })
                resp = api.upload_datafile(
                    args.doi, join(root,f), df.json())
                print(f"Uploaded: {join(root, f)} — Status: {resp.status_code}")
                check_dataset_lock(5)

    # Extract and modify the citation block
    full_metadata = dataset.json()["data"]["latestVersion"]["metadataBlocks"]
    citation_block = full_metadata["citation"]

    # Update the title field
    for field in citation_block["fields"]:
        if field["typeName"] == "title":
            field["value"] = args.title

    # Construct full metadata payload
    updated_metadata = {
        "metadataBlocks": {
            "citation": citation_block
        }
    }

    # Build PUT request
    headers = {
        "Content-Type": "application/json",
        "X-Dataverse-key": token
    }
    url = f"{dataverse_server}/api/datasets/:persistentId/versions/:draft"
    params = {
        "persistentId": args.doi,
        "replace": "true"
    }
    resp = requests.put(url, headers=headers, params=params, data=json.dumps(updated_metadata))
    print("Metadata update response code:", resp.status_code)
    print("Metadata update response body:", resp.text)

    if resp.status_code != 200:
        raise Exception("Failed to update metadata.")

    if args.publish.lower() == 'true':
        # publish updated dataset
        resp = api.publish_dataset(args.doi, release_type="major")