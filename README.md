# The AI judge repo where the magic happens

Content:

- Scope
- Data
- Physical systems
- Papers & Models
- Installation guide

## Main focus (MVP)

Focus = DD3. (Double Dutch Single Freestyle)

1) Jumper Localisation (CNN based)
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
