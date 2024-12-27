# Browse and navigates storage device, to find and discover videos.

# Browse and navigate storage to find DB orphans

# Nice to have : Browse and navigate storage to find renames that happend on the drive

## Legacy code

    # # Read from the most inner nested to outside
    # return list(
    #     map(
    #         lambda vid : {
    #             "title" : vid,
    #             "image-preview" : os.path.join(API_URL, "videoimage", subFolder, urllib.parse.quote(vid))
    #         },
    #         # Map file string title to
    #         filter(
    #             # Filter supported formats e.g. ".mp4"
    #             lambda f : f.split('.')[-1].lower() in SUPPORTED_VIDEO_FORMATS,
    #             filter(
    #                 # Filter files from files and subfolders
    #                 lambda content : os.path.isfile(os.path.join(VIDEO_FOLDER, subFolder, content)), 
    #                 os.listdir(os.path.join(VIDEO_FOLDER, subFolder))
    #             )
    #         )
    #     )
    # )