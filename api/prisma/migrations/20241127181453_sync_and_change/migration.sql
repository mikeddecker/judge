-- CreateTable
CREATE TABLE `BorderLabels` (
    `id` INTEGER NOT NULL AUTO_INCREMENT,
    `videoID` INTEGER NOT NULL,
    `frameNr` SMALLINT NOT NULL,
    `label` TINYINT NOT NULL,
    `manual_insert` BOOLEAN NOT NULL,

    INDEX `videoID`(`videoID`),
    PRIMARY KEY (`id`)
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- CreateTable
CREATE TABLE `Folders` (
    `folderID` INTEGER NOT NULL AUTO_INCREMENT,
    `path` VARCHAR(127) NOT NULL,

    PRIMARY KEY (`folderID`)
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- CreateTable
CREATE TABLE `FrameLabels` (
    `id` INTEGER NOT NULL AUTO_INCREMENT,
    `videoID` INTEGER NOT NULL,
    `frameNr` SMALLINT NOT NULL,
    `label` TINYINT NOT NULL,
    `manual_insert` BOOLEAN NOT NULL,
    `rect_center_x` FLOAT NULL,
    `rect_center_y` FLOAT NULL,
    `rect_size` FLOAT NULL,

    INDEX `videoID`(`videoID`),
    PRIMARY KEY (`id`)
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- CreateTable
CREATE TABLE `Videos` (
    `videoID` INTEGER NOT NULL AUTO_INCREMENT,
    `folderID` INTEGER NOT NULL,
    `name` VARCHAR(255) NOT NULL,
    `width` SMALLINT NOT NULL,
    `height` SMALLINT NOT NULL,
    `fps` TINYINT NOT NULL,
    `training` BOOLEAN NOT NULL,
    `quality` TINYINT NOT NULL,
    `obstruction` BOOLEAN NOT NULL,

    INDEX `folderID`(`folderID`),
    PRIMARY KEY (`videoID`)
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- AddForeignKey
ALTER TABLE `BorderLabels` ADD CONSTRAINT `BorderLabels_ibfk_1` FOREIGN KEY (`videoID`) REFERENCES `Videos`(`videoID`) ON DELETE NO ACTION ON UPDATE NO ACTION;

-- AddForeignKey
ALTER TABLE `FrameLabels` ADD CONSTRAINT `FrameLabels_ibfk_1` FOREIGN KEY (`videoID`) REFERENCES `Videos`(`videoID`) ON DELETE NO ACTION ON UPDATE NO ACTION;

-- AddForeignKey
ALTER TABLE `Videos` ADD CONSTRAINT `Videos_ibfk_1` FOREIGN KEY (`folderID`) REFERENCES `Folders`(`folderID`) ON DELETE NO ACTION ON UPDATE NO ACTION;
