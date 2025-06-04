# The AI judge assistent repo

Note that, in the last couple of weeks some adaptions here and there have taken place, so not all flows may work.

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

Segmenting each individual action so that it can be recognized.
Currently, only uses the [Multiscale Video Transformer](https://docs.pytorch.org/vision/main/models/video_mvit.html)

### Recognition

Eventuall, each section can be transformed into a (1, 3, 16, 224, 224) input, (batch_size, channels, timesteps, height, width) or (B, C, T, H, W).
![image](https://github.com/user-attachments/assets/f36e7ed3-f5ce-4566-96a6-4abd0a25b491)

For full details see [paper](./paper/bachelorproef/DeDeckerMikeBP.tex), preferably compile it from a [pdf](./paper/DeDeckerMikeBP.tex)

### Double Dutch Single (DD3) Data

(private)
Freestyles: 450+, (as competed in national e.g. [Belgium](gymfed.be) or internationally [IJRU](https://ijru.sport/))
Labeled: 50+ (1 hour)

### Physical devices for training

Laptop Acer Nitro ANV15-51, running Ubuntu 24.04.2 LTS, using a 13th Gen Intel® Core™ i5-13420H × 12, with 16GB RAM and a
NVIDIA GeForce RTX™ 4050 Laptop GPU 5898MiB.

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
