CREATE OR REPLACE VIEW FrameLabelsTeam AS
SELECT
	videoId,
	frameNr,
	0 as labeltype,
	MIN(x - width / 2) AS xmin,
	MAX(x + width / 2) AS xmax,
	MIN(y - height / 2) AS ymin,
	MAX(y + height / 2) AS ymax,
	MAX(x + width / 2) - MIN(x - width / 2) AS width,
	(MAX(x + width / 2) + MIN(x - width / 2)) / 2 AS x,
	MAX(y + height / 2) - MIN(y - height / 2) AS height,
	(MAX(y + height / 2) + MIN(y - height / 2)) / 2 AS y    
FROM FrameLabels
WHERE labeltype = 1
GROUP BY videoId, frameNr
ORDER BY videoId, frameNr;

CREATE OR REPLACE VIEW FrameLabelsExtended AS
SELECT
	videoId,
	frameNr,
	labeltype,
	x - width / 2 AS xmin,
	x + width / 2 AS xmax,
	y - height / 2 AS ymin,
	y + height / 2 AS ymax,
	x, y, width, height
FROM FrameLabels
ORDER BY videoId, frameNr;

CREATE OR REPLACE VIEW FrameLabelsAll AS
SELECT
	videoId, frameNr, labeltype, v.training,
    xmin, xmax, ymin, ymax,
    x, y, flt.width, flt.height
FROM FrameLabelsTeam flt
JOIN Videos v ON flt.videoId = v.id
UNION
SELECT
	videoId, frameNr, labeltype, v.training,
    xmin, xmax, ymin, ymax,
    x, y, fle.width, fle.height
FROM FrameLabelsExtended fle
JOIN Videos v ON fle.videoId = v.id
ORDER BY videoId, frameNr, labeltype;

