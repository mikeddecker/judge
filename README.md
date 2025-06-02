# The AI judge repo where the magic happens

Content:

- Scope
- Data
- Physical systems
- Papers & Models
- Installation guide

## Main focus (MVP)

Focus = DD3. (Double Dutch Single Freestyle)

### Jumper Localisation

Jumper localization crops the athletes, frame by frame in a full video. 
This uses [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics), licensed under AGPL-3.0.

### Segmentation



### Recognition

Eventuall, each section can be transformed into a (1, 3, 16, 224, 224) input, (batch_size, channels, timesteps, height, width) or (B, C, T, H, W).
![image](https://github.com/user-attachments/assets/f36e7ed3-f5ce-4566-96a6-4abd0a25b491)



2) Skill segmentation, start of skill, end of skill (Using something like LTContext or others using [papers-with-code](https://paperswithcode.com/task/action-segmentation))
3) Counting rotations (DU, TU, QU, Wrap with 3 rotations?) (ConvLSTM, MiM, (Vision)transformer)
4) ... (unknown in betweens)
5) Label the effective skill (ConvLSTM, MiM, (Vision)transformer)

For full details see [proposal](./paper/voorstel/voorstel-inhoud.tex), preferably compile it from [full-proposal](./paper/voorstel/DeDeckerMike-BPvoorstel.tex)

### DD3 Data

Current estimation of data: 440GB

Includes mostly teams: SR2, SR4, DD3, DD4, already \
Some SR freestyles.

Included:

- Belgium 2022 Big Nationals
- Belgium 2023 Team Nationals
- Belgium 2024 Team Provincials A. (partially/from tablet)
- Belgium 2024 Team Nationals
- Belgium 2024 Team provincials B.
- Some freestyles

TODO : scrape YT subscribed channels (clubs, virtual world championships, WC livestreams...)
TODO : scrape insta videos (=mainly SR)

### Physical devices for training

Laptop: GPU (not ideal, but good for try-outs)
School: Ask for server (with GPU) --> probably not.

### Resources (Papers & Models & additional sources)

Resources becomes a folder/file like paperlinks.md to store thoughts and summaries about papers. (The thinking process.) \
[paperlinks](administratie/paperlinks.md)

### Installation guide

Install requirements

`pip3 install -r requirements.txt`

Then start up the API and the web app.

- [API](./api/README.md) (Includes docker)
- [Web](./web/README.md)

### BACKUP

```bash
mysqldump -h 127.0.0.1 -P 3377 -u root -p judge > "/media/miked/Elements/Judge/FINISHED-DB-READY/$(date +\%Y\%m\%d)_judge_dump.sql"
```
