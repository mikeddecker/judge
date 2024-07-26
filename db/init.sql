CREATE DATABASE IF NOT EXISTS judge;

DROP TABLE IF EXISTS FrameLabels;
DROP TABLE IF EXISTS BorderLabels;
DROP TABLE IF EXISTS Borders;
DROP TABLE IF EXISTS Videos;
DROP TABLE IF EXISTS Folders;

CREATE TABLE Folders (
    folderID INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    path NVARCHAR(50) NOT NULL
);

CREATE TABLE Videos (
    videoID INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    folderID INT NOT NULL,
    name NVARCHAR(255) NOT NULL,
    training BOOLEAN NOT NULL,
    obstruction TINYINT NOT NULL, -- 0 geen - 100 high obstruction
    quality TINYINT NOT NULL, -- 0 zeer wazig - 100 goed
    type TINYINT NOT NULL, -- 0 = competitie, 1 = distant, sporthal-like, 2 = dicht & vrij, 3 = distant, sporthallike, ruis, 4 = dicht & achtergrondruis
    dimensionX SMALLINT NOT NULL, -- 1920 of 1080
    dimensionY SMALLINT NOT NULL, -- 1080 of 1920 of 3k...

    FOREIGN KEY (folderID) REFERENCES Folders(folderID)
);

CREATE TABLE FrameLabels ( -- CNN based
    videoID INT NOT NULL,
    frameNr SMALLINT NOT NULL, -- max 32k
    label TINYINT NOT NULL, -- 0 - 9 currently (on ground, air, release, power...)
    manual_insert BOOLEAN NOT NULL, -- 0 not, 1 manual
    rect_center_x FLOAT(3) DEFAULT NULL,
    rect_center_y FLOAT(3) DEFAULT NULL,
    rect_size FLOAT(3) DEFAULT NULL,
    
    FOREIGN KEY (videoID) REFERENCES Videos(videoID)
);

CREATE TABLE BorderLabels ( -- Interval based
    videoID INT NOT NULL,
    frameNr SMALLINT NOT NULL,
    label TINYINT NOT NULL, -- 0 no skill, 1 start, 2 executing skill, 3 skill end
    manual_insert BOOLEAN NOT NULL, -- 0 not, 1 manual
    
    FOREIGN KEY (videoID) REFERENCES Videos(videoID)
);


DELIMITER $$

/**
    Procedure: Frame batch_nrs order
    e.g.
    1) video_id 2, batch 5
    2) video_id 8, batch 66
    3) video_id 3, batch 27
    4) video_id 1, batch 21
    5) video_id 2, batch 66
    ...
*/

DROP PROCEDURE IF EXISTS GetFrameBatchNrs$$
CREATE PROCEDURE GetFrameBatchNrs (IN batch_size INT, IN train BOOLEAN)
BEGIN
	WITH train_or_val_idxs AS (
		SELECT videoID FROM Videos WHERE training = train
    ), frame_batch_nrs AS (
		SELECT *, FLOOR((frameNr-1) / 16) as batch_nr_video FROM FrameLabels 
        WHERE videoID IN (SELECT * FROM train_or_val_idxs) AND rect_center_x IS NOT NULL
	)
	SELECT videoID, batch_nr_video, ROW_NUMBER() OVER (ORDER BY RAND()) - 1 AS batch_id
	FROM frame_batch_nrs 
    GROUP by videoID, batch_nr_video
    ORDER BY batch_id;
END$$


/**
    Procedure: getting the i'th batch
    VideoID = 2, batch_nr = 3, batch_size = 16
    --> frameLabels list from [33, 48] 
*/
DROP PROCEDURE IF EXISTS GetRectLabels$$
CREATE PROCEDURE GetRectLabels(video_id INT, batch_nr INT, batch_size INT)
BEGIN
	SELECT * FROM FrameLabels 
    WHERE videoID = video_id AND FrameNr BETWEEN batch_nr * batch_size + 1 AND (batch_nr + 1) * batch_size AND rect_center_x IS NOT NULL
    ORDER BY FrameNr;
END$$

DELIMITER ;