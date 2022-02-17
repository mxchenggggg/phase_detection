import os

from office365.runtime.auth.user_credential import UserCredential
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file_system_object_type import FileSystemObjectType

site_url = "https://livejohnshopkins.sharepoint.com/sites/SurgicalPhaseSegmentation/"
username = "xma42@jh.edu"
password = "619223Mxc!robotics"

abs_file_url = site_url + "Shared Documents/test.txt"
abs_file_url = site_url + "Shared Documents/Milestones.xlsx"


test_client_credentials = UserCredential(username, password)

ctx = ClientContext(site_url).with_credentials(test_client_credentials)



doc_lib = ctx.web.lists.get_by_title("Documents")
items = doc_lib.items.select(["FileSystemObjectType"]).expand(["File", "Folder"]).get().execute_query()
for item in items:  # type: ListItem
    if item.file_system_object_type == FileSystemObjectType.Folder:
        print("Folder url: {0}".format(item.folder.serverRelativeUrl))
    else:
        print("File url: {0}".format(item.file.serverRelativeUrl))


file_url = "/sites/SurgicalPhaseSegmentation/Shared Documents/test.txt"
download_path = os.path.join("./", os.path.basename(file_url))
with open(download_path, "wb") as local_file:
    file = ctx.web.get_file_by_server_relative_path(file_url).download(local_file).execute_query()
print("[Ok] file has been downloaded into: {0}".format(download_path))


path = "./testupload.txt"
with open(path, 'rb') as content_file:
    file_content = content_file.read()

list_title = "Documents"
target_folder = ctx.web.lists.get_by_title(list_title).root_folder
name = os.path.basename(path)
target_file = target_folder.upload_file(name, file_content).execute_query()
print("File has been uploaded to url: {0}".format(target_file.serverRelativeUrl))