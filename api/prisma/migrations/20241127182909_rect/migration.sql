/*
  Warnings:

  - You are about to drop the column `rect_size` on the `FrameLabels` table. All the data in the column will be lost.
  - Added the required column `rect_size_x` to the `FrameLabels` table without a default value. This is not possible if the table is not empty.
  - Added the required column `rect_size_y` to the `FrameLabels` table without a default value. This is not possible if the table is not empty.
  - Made the column `rect_center_x` on table `FrameLabels` required. This step will fail if there are existing NULL values in that column.
  - Made the column `rect_center_y` on table `FrameLabels` required. This step will fail if there are existing NULL values in that column.

*/
-- AlterTable
ALTER TABLE `FrameLabels` DROP COLUMN `rect_size`,
    ADD COLUMN `rect_size_x` FLOAT NOT NULL,
    ADD COLUMN `rect_size_y` FLOAT NOT NULL,
    MODIFY `rect_center_x` FLOAT NOT NULL,
    MODIFY `rect_center_y` FLOAT NOT NULL;
