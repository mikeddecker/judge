USE judge;

CREATE OR REPLACE VIEW SkillTrainDistribution AS
SELECT sds.name, COUNT(sd.skill) as skill_count,
COUNT(sd.skill) * 100 /c.total as total
FROM Skillinfo_DoubleDutch sd
JOIN Skillinfo_DoubleDutch_Skill sds ON sd.skill = sds.id
JOIN (SELECT COUNT(*) as total FROM Skillinfo_DoubleDutch WHERE MOD(videoId, 10) <> 5) c
WHERE MOD(videoId, 10) <> 5
GROUP BY sd.skill, sds.name, c.total
ORDER BY skill_count DESC;

CREATE OR REPLACE VIEW SkillValDistribution AS
SELECT sds.name, COUNT(sd.skill) as skill_count,
COUNT(sd.skill) * 100 /c.total as total
FROM Skillinfo_DoubleDutch sd
JOIN Skillinfo_DoubleDutch_Skill sds ON sd.skill = sds.id
JOIN (SELECT COUNT(*) as total FROM Skillinfo_DoubleDutch WHERE MOD(videoId, 10) = 5) c
WHERE MOD(videoId, 10) = 5
GROUP BY sd.skill, sds.name, c.total
ORDER BY skill_count DESC;

CREATE OR REPLACE VIEW SkillTrainTestComparison AS
SELECT t.total as train_percentage, t.skill_count as train_count, t.name as skillname, v.skill_count as val_percentage, v.total as val_total
FROM SkillTrainDistribution t
JOIN SkillValDistribution v
ON t.name = v.name;

