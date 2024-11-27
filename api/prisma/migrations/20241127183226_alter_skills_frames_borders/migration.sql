/*
  Warnings:

  - You are about to drop the `BorderLabels` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `FrameLabels` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropForeignKey
ALTER TABLE `BorderLabels` DROP FOREIGN KEY `BorderLabels_ibfk_1`;

-- DropForeignKey
ALTER TABLE `FrameLabels` DROP FOREIGN KEY `FrameLabels_ibfk_1`;

-- DropTable
DROP TABLE `BorderLabels`;

-- DropTable
DROP TABLE `FrameLabels`;

-- CreateTable
CREATE TABLE `Skillsegment` (
    `id` INTEGER NOT NULL AUTO_INCREMENT,
    `videoID` INTEGER NOT NULL,
    `frameStart` SMALLINT NOT NULL,
    `frameEnd` SMALLINT NOT NULL,
    `label` TINYINT NOT NULL,
    `manual_insert` BOOLEAN NOT NULL,

    INDEX `videoID`(`videoID`),
    PRIMARY KEY (`id`)
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- CreateTable
CREATE TABLE `FrameLabel` (
    `id` INTEGER NOT NULL AUTO_INCREMENT,
    `videoID` INTEGER NOT NULL,
    `frameNr` SMALLINT NOT NULL,
    `label` TINYINT NOT NULL,
    `rect_center_x` FLOAT NOT NULL,
    `rect_center_y` FLOAT NOT NULL,
    `rect_size_x` FLOAT NOT NULL,
    `rect_size_y` FLOAT NOT NULL,

    INDEX `videoID`(`videoID`),
    PRIMARY KEY (`id`)
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- AddForeignKey
ALTER TABLE `Skillsegment` ADD CONSTRAINT `BorderLabels_ibfk_1` FOREIGN KEY (`videoID`) REFERENCES `Videos`(`videoID`) ON DELETE NO ACTION ON UPDATE NO ACTION;

-- AddForeignKey
ALTER TABLE `FrameLabel` ADD CONSTRAINT `FrameLabels_ibfk_1` FOREIGN KEY (`videoID`) REFERENCES `Videos`(`videoID`) ON DELETE NO ACTION ON UPDATE NO ACTION;
