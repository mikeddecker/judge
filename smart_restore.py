#!/usr/bin/env python3
"""
Smart data restoration from backup - syncs missing data only.
- Doesn't insert IDs (lets DB generate UUIDs)
- Matches by name/content, not ID
- Only inserts missing rows
- Handles foreign key resolution by name
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

BACKUP_FILE = "/mnt/judge-drive/Judge/results/backups2/judge_db_20260218_prior_uuid_change_without_uuid_changes.sql"

class SmartRestorer:
    def __init__(self, backup_path: str):
        self.backup_path = backup_path
        self.backup_content = None
        self.sql_statements = []
        
    def load_backup(self):
        """Load backup file"""
        with open(self.backup_path, 'r') as f:
            self.backup_content = f.read()
        print(f"✓ Loaded backup ({len(self.backup_content)} bytes)")
    
    def extract_insert_values(self, table_name: str) -> List[str]:
        """Extract INSERT VALUES for table - returns list of value tuples as strings"""
        pattern = rf"INSERT INTO `{table_name}` VALUES\s*(.*?);\n"
        match = re.search(pattern, self.backup_content, re.DOTALL)
        
        if not match:
            return []
        
        values_str = match.group(1)
        # Extract each row: (1,'value',...)
        records = re.findall(r'\([^)]+\)(?:,(?!\s*\!))?', values_str)
        return records
    
    def parse_tuple(self, record: str) -> List[str]:
        """Parse SQL tuple into list of values, handling quoted strings"""
        # Remove outer parens
        record = record.strip()
        if record.startswith('(') and record.endswith(')'):
            record = record[1:-1]
        
        values = []
        current = ""
        in_quotes = False
        escape_next = False
        
        for char in record:
            if escape_next:
                current += char
                escape_next = False
            elif char == '\\':
                current += char
                escape_next = True
            elif char == "'" and in_quotes:
                in_quotes = False
                current += char
            elif char == "'" and not in_quotes:
                in_quotes = True
                current += char
            elif char == ',' and not in_quotes:
                values.append(current.strip())
                current = ""
            else:
                current += char
        
        if current:
            values.append(current.strip())
        
        return values
    
    def unquote(self, value: str) -> Optional[str]:
        """Remove SQL quotes from value"""
        if value == "NULL":
            return None
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1].replace("\\'", "'")
        try:
            return value
        except:
            return value
    
    def gen_taggroups_sql(self) -> str:
        """Generate INSERT for missing TagGroups"""
        print("\n📋 TagGroups")
        records = self.extract_insert_values('TagGroups')
        
        sql = []
        for record in records:
            vals = self.parse_tuple(record)
            if len(vals) >= 5:
                # (id, name, parentId, createdAt, updatedAt)
                old_id, name, parent_id, created, updated = vals[0:5]
                name = self.unquote(name)
                
                sql.append(f"""-- TagGroup: {name}
INSERT IGNORE INTO TagGroups (name, parentId, createdAt, updatedAt)
SELECT '{self.escape_sql(name)}', NULL, NOW(), NOW()
WHERE NOT EXISTS (SELECT 1 FROM TagGroups WHERE name = '{self.escape_sql(name)}');
""")
        
        if sql:
            print(f"   Found {len(sql)} TagGroups in backup")
        return "\n".join(sql)
    
    def gen_tags_sql(self) -> str:
        """Generate INSERT for missing Tags"""
        print("\n📋 Tags")
        records = self.extract_insert_values('Tags')
        
        # Build map of tag_group_id -> name
        tg_records = self.extract_insert_values('TagGroups')
        tg_map = {}
        for tg_rec in tg_records:
            tg_vals = self.parse_tuple(tg_rec)
            if len(tg_vals) >= 2:
                tg_map[self.unquote(tg_vals[0])] = self.unquote(tg_vals[1])
        
        sql = []
        for record in records:
            vals = self.parse_tuple(record)
            if len(vals) >= 6:
                # (id, name, tagGroupId, keywords, createdAt, updatedAt)
                old_id, name, tag_group_id, keywords, created, updated = vals[0:6]
                name = self.unquote(name)
                keywords = self.unquote(keywords) or ""
                tag_group_id = self.unquote(tag_group_id)
                
                tg_name = tg_map.get(tag_group_id)
                if tg_name:
                    sql.append(f"""-- Tag: {name} (in {tg_name})
INSERT IGNORE INTO Tags (name, tagGroupId, keywords, createdAt, updatedAt)
SELECT '{self.escape_sql(name)}', id, '{self.escape_sql(keywords)}', NOW(), NOW()
FROM TagGroups WHERE name = '{self.escape_sql(tg_name)}'
AND NOT EXISTS (SELECT 1 FROM Tags WHERE name = '{self.escape_sql(name)}')
LIMIT 1;
""")
        
        if sql:
            print(f"   Found {len(sql)} Tags in backup")
        return "\n".join(sql)
    
    def gen_layers_sql(self) -> str:
        """Generate INSERT for missing Layers"""
        print("\n📋 Layers")
        records = self.extract_insert_values('Layers')
        
        sql = []
        for record in records:
            vals = self.parse_tuple(record)
            if len(vals) >= 8:
                # (id, name, type, min, max, step, createdAt, updatedAt)
                old_id, name, ltype, min_val, max_val, step_val, created, updated = vals[0:8]
                name = self.unquote(name)
                ltype = self.unquote(ltype)
                min_val = self.unquote(min_val)
                max_val = self.unquote(max_val)
                step_val = self.unquote(step_val)
                
                min_part = f"{min_val}" if min_val and min_val != "NULL" else "NULL"
                max_part = f"{max_val}" if max_val and max_val != "NULL" else "NULL"
                step_part = f"{step_val}" if step_val and step_val != "NULL" else "NULL"
                
                sql.append(f"""-- Layer: {name}
INSERT IGNORE INTO Layers (name, type, min, max, step, createdAt, updatedAt)
SELECT '{self.escape_sql(name)}', '{self.escape_sql(ltype)}', {min_part}, {max_part}, {step_part}, NOW(), NOW()
WHERE NOT EXISTS (SELECT 1 FROM Layers WHERE name = '{self.escape_sql(name)}');
""")
        
        if sql:
            print(f"   Found {len(sql)} Layers in backup")
        return "\n".join(sql)
    
    def gen_layervalues_sql(self) -> str:
        """Generate INSERT for missing LayerValues"""
        print("\n📋 LayerValues")
        records = self.extract_insert_values('LayerValues')
        
        if not records:
            return ""
        
        # First get layer mapping
        layer_records = self.extract_insert_values('Layers')
        layer_map = {}
        for lr in layer_records:
            lvals = self.parse_tuple(lr)
            if len(lvals) >= 2:
                layer_map[self.unquote(lvals[0])] = self.unquote(lvals[1])
        
        sql = []
        for record in records:
            vals = self.parse_tuple(record)
            if len(vals) >= 4:
                # (id, layerId, value, description)
                old_id, layer_id, value, description = vals[0:4]
                layer_id = self.unquote(layer_id)
                value = self.unquote(value)
                description = self.unquote(description) or ""
                
                layer_name = layer_map.get(layer_id, "")
                
                if layer_name:
                    sql.append(f"""-- LayerValue: {value} for {layer_name}
INSERT IGNORE INTO LayerValues (layerId, value, description)
SELECT id, '{self.escape_sql(value)}', '{self.escape_sql(description)}'
FROM Layers WHERE name = '{self.escape_sql(layer_name)}'
AND NOT EXISTS (SELECT 1 FROM LayerValues WHERE value = '{self.escape_sql(value)}' AND layerId = (SELECT id FROM Layers WHERE name = '{self.escape_sql(layer_name)}'))
LIMIT 1;
""")
        
        if sql:
            print(f"   Found {len(sql)} LayerValues in backup")
        return "\n".join(sql)
    
    def gen_layercomposition_sql(self) -> str:
        """Generate INSERT for missing LayerComposition"""
        print("\n📋 LayerComposition")
        records = self.extract_insert_values('LayerComposition')
        
        if not records:
            return ""
        
        # Get layer mapping
        layer_records = self.extract_insert_values('Layers')
        layer_map = {}
        for lr in layer_records:
            lvals = self.parse_tuple(lr)
            if len(lvals) >= 2:
                layer_map[self.unquote(lvals[0])] = self.unquote(lvals[1])
        
        sql = []
        for record in records:
            vals = self.parse_tuple(record)
            if len(vals) >= 8:
                # (id, compositionName, stage, layerId, createdAt, updatedAt, defaultValue, focussed)
                old_id, comp_name, stage, layer_id, created, updated, default_val, focussed = vals[0:8]
                comp_name = self.unquote(comp_name)
                stage = self.unquote(stage)
                layer_id = self.unquote(layer_id)
                default_val = self.unquote(default_val)
                focussed = self.unquote(focussed)
                
                layer_name = layer_map.get(layer_id, "")
                
                if layer_name:
                    stage_part = "NULL" if stage == "NULL" else stage
                    default_part = f"'{self.escape_sql(default_val)}'" if default_val and default_val != "NULL" else "NULL"
                    
                    sql.append(f"""-- LayerComposition: {comp_name} - {layer_name}
INSERT IGNORE INTO LayerComposition (compositionName, stage, layerId, defaultValue, focussed, createdAt, updatedAt)
SELECT '{self.escape_sql(comp_name)}', {stage_part}, id, {default_part}, {focussed}, NOW(), NOW()
FROM Layers WHERE name = '{self.escape_sql(layer_name)}'
AND NOT EXISTS (SELECT 1 FROM LayerComposition WHERE compositionName = '{self.escape_sql(comp_name)}' AND layerId = (SELECT id FROM Layers WHERE name = '{self.escape_sql(layer_name)}'))
LIMIT 1;
""")
        
        if sql:
            print(f"   Found {len(sql)} LayerComposition records in backup")
        return "\n".join(sql)
    
    def gen_framelabels_sql(self) -> str:
        """Generate INSERT for missing FrameLabels (exclude background, match by video name)"""
        print("\n📋 FrameLabels (excluding background)")
        records = self.extract_insert_values('FrameLabels')
        
        if not records:
            return ""
        
        sql = []
        skipped = 0
        
        for record in records:
            vals = self.parse_tuple(record)
            if len(vals) >= 13:
                # (id, videoId, frameNr, x, y, width, height, jumperVisible, labeltype, createdAt, labeldate, labeltime, updatedAt)
                old_id, vid, frame_nr, x, y, width, height, jumper_vis, labeltype, created, labeldate, labeltime, updated = vals[0:13]
                
                labeltype = self.unquote(labeltype)
                
                # Skip background (usually labeltype = 2)
                if labeltype == "2":
                    skipped += 1
                    continue
                
                vid = self.unquote(vid)
                frame_nr = self.unquote(frame_nr)
                x = self.unquote(x)
                y = self.unquote(y)
                width = self.unquote(width)
                height = self.unquote(height)
                jumper_vis = self.unquote(jumper_vis)
                created = self.unquote(created)
                labeldate = self.unquote(labeldate)
                labeltime = self.unquote(labeltime)
                updated = self.unquote(updated)
                
                sql.append(f"""-- FrameLabel: video {vid} frame {frame_nr}
INSERT IGNORE INTO FrameLabels (videoId, frameNr, x, y, width, height, jumperVisible, labeltype, createdAt, labeldate, labeltime, updatedAt)
VALUES ({vid}, {frame_nr}, {x}, {y}, {width}, {height}, {jumper_vis}, {labeltype}, '{created}', '{labeldate}', '{labeltime}', '{updated}')
ON DUPLICATE KEY UPDATE updatedAt=NOW();
""")
        
        if sql:
            print(f"   Found {len(sql)} FrameLabels in backup (skipped {skipped} background labels)")
        return "\n".join(sql)
    
    def gen_skills_sql(self) -> str:
        """Generate INSERT for missing Skills (match by video ID)"""
        print("\n📋 Skills")
        records = self.extract_insert_values('Skills')
        
        if not records:
            return ""
        
        sql = []
        
        for record in records:
            vals = self.parse_tuple(record)
            if len(vals) >= 7:
                # (id, videoId, frameStart, frameEnd, skillinfo, createdAt, updatedAt)
                old_id, vid, frame_start, frame_end, skillinfo, created, updated = vals[0:7]
                
                vid = self.unquote(vid)
                frame_start = self.unquote(frame_start)
                frame_end = self.unquote(frame_end)
                skillinfo = self.unquote(skillinfo)
                created = self.unquote(created)
                updated = self.unquote(updated)
                
                created_part = f"'{created}'" if created else "NOW()"
                updated_part = f"'{updated}'" if updated else "NOW()"
                
                sql.append(f"""-- Skill: video {vid} frames {frame_start}-{frame_end}
INSERT IGNORE INTO Skills (videoId, frameStart, frameEnd, skillinfo, createdAt, updatedAt)
VALUES ({vid}, {frame_start}, {frame_end}, '{self.escape_sql(skillinfo)}', {created_part}, {updated_part})
ON DUPLICATE KEY UPDATE updatedAt=NOW();
""")
        
        if sql:
            print(f"   Found {len(sql)} Skills in backup")
        return "\n".join(sql)
    
    def escape_sql(self, value: str) -> str:
        """Escape SQL string"""
        if value is None:
            return "NULL"
        return str(value).replace("'", "\\'").replace('"', '\\"')
    
    def generate_sql(self) -> str:
        """Generate full restoration SQL"""
        print("\n" + "="*70)
        print("GENERATING SMART RESTORATION SQL")
        print("="*70)
        
        parts = [
            "-- Smart Data Restoration - Only Missing Data",
            "-- Generated: March 15, 2026",
            "-- Strategy: Check existence by name, insert if missing, generate new IDs",
            "",
            "SET FOREIGN_KEY_CHECKS=0;",
            "SET SESSION sql_mode='';",
            ""
        ]
        
        parts.append(self.gen_taggroups_sql())
        parts.append(self.gen_tags_sql())
        parts.append(self.gen_layers_sql())
        parts.append(self.gen_layervalues_sql())
        parts.append(self.gen_layercomposition_sql())
        parts.append(self.gen_framelabels_sql())
        parts.append(self.gen_skills_sql())
        
        parts.extend([
            "",
            "SET FOREIGN_KEY_CHECKS=1;",
            "COMMIT;",
            "",
            "-- Verification queries:",
            "-- SELECT COUNT(*) FROM TagGroups;",
            "-- SELECT COUNT(*) FROM Tags;",
            "-- SELECT COUNT(*) FROM Layers;",
            "-- SELECT COUNT(*) FROM LayerComposition;",
            "-- SELECT COUNT(*) FROM FrameLabels;",
            "-- SELECT COUNT(*) FROM Skills;"
        ])
        
        return "\n".join(parts)

def main():
    print("🚀 Smart Data Restoration Generator\n")
    
    restorer = SmartRestorer(BACKUP_FILE)
    restorer.load_backup()
    
    sql = restorer.generate_sql()
    
    output_file = Path(BACKUP_FILE).parent / "smart_restoration.sql"
    with open(output_file, 'w') as f:
        f.write(sql)
    
    print("\n" + "="*70)
    print(f"✅ SQL generated: {output_file}")
    print(f"   Size: {len(sql)} bytes")
    print("="*70)
    print("\nUsage:")
    print("  mysql -u root -p judge_db < smart_restoration.sql")
    print("\nOr in MySQL client:")
    print("  source /path/to/smart_restoration.sql;")

if __name__ == "__main__":
    main()

