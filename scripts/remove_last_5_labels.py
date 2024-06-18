from utils_misc import pickle_save, pickle_load_or_create

video_border_labels_path = 'df_video_border_labels_seq'

df_labels = pickle_load_or_create(video_border_labels_path, lambda x: {})

pickle_save(video_border_labels_path, df_labels.loc[:len(df_labels)-5])
