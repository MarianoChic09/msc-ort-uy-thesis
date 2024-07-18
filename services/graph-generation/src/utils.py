def save_uploaded_file(upload_file, destination_path):
    with open(destination_path, "wb+") as file_object:
        file_object.write(upload_file.file.read())
