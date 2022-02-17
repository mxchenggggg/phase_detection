import os

from office365.runtime.auth.user_credential import UserCredential
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file_system_object_type import FileSystemObjectType

site_url = "https://livejohnshopkins.sharepoint.com/sites/SurgicalPhaseSegmentation/"
shared_doc_url = "/sites/SurgicalPhaseSegmentation/Shared Documents/"


class SharePointSite(object):
    
    def __init__(self, JHEDID=None, password=None):
        
        JHEDID = JHEDID or input("Enter your JHEDID: ")
        password = password or input("Enter your password: ")
        user_name = JHEDID + "@jh.edu"

        self.ctx = ClientContext(site_url).with_credentials(
            UserCredential(user_name, password))

    def download(self, source, target_dir):
        source_file_url = os.path.join(shared_doc_url, source)
        target = os.path.join(target_dir, os.path.basename(source))

        with open(target, "wb") as target_file:
            file = self.ctx.web.get_file_by_server_relative_path(
                source_file_url).download(target_file).execute_query()
        print("Downloaded {0} from SharePoint into: {1}".format(
            source, target))

    def upload(self, source, target_dir=""):
        target_url = os.path.join(shared_doc_url, target_dir)
        target_file_name = os.path.basename(source)

        with open(source, 'rb') as content_file:
            file_content = content_file.read()

        file = self.ctx.web.get_folder_by_server_relative_url(
            target_url).upload_file(target_file_name, file_content).execute_query()
        print("{0} has been uploaded to: {1}".format(source, file.serverRelativeUrl))

    def list_dirs_files(self):
        doc_lib = self.ctx.web.lists.get_by_title("Documents")
        items = doc_lib.items.select(["FileSystemObjectType"]).expand(
            ["File", "Folder"]).get().execute_query()
        for item in items:  # type: ListItem
            if item.file_system_object_type == FileSystemObjectType.Folder:
                print("Folder url: {0}".format(item.folder.serverRelativeUrl))
            else:
                print("File url: {0}".format(item.file.serverRelativeUrl))


JHEDID = input("Enter your JHEDID: ")
password = input("Enter your password: ")

site = SharePointSite(JHEDID, password)
site.list_dirs_files()
site.download("test.txt", "./")
site.upload("./testupload.txt")
