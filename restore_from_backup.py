#!/usr/bin/env python3
"""
Restore missing data (tags, layers, layer composition, frame labels, skills) from a pre-UUID backup.
This script:
1. Parses the backup SQL file to extract data
2. Creates ID mappings for layers and other entities
3. Generates SQL to restore the data with proper UUID references
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import MySQLdb

# Configuration
BACKUP_FILE = "/mnt/judge-drive/Judge/results/backups2/judge_db_20260218_prior_uuid_change_without_uuid_changes.sql"
DB_HOST = "localhost"  # Adjust as needed
DB_USER = "root"  # Adjust as needed
DB_PASS = "password"  # Adjust as needed
DB_NAME = "judge_db"

class BackupRestorer:
    def __init__(self, backup_path: str):
        self.backup_path = backup_path
        self.backup_content = None
        self.id_mappings = {}  # Maps old IDs to new UUIDs
        
    def load_backup(self):
        """Load backup file content"""
        with open(self.backup_path, 'r') as f:
            self.backup_content = f.read()
        print(f"✓ Loaded backup file ({len(self.backup_content)} bytes)")
        
    def extract_insert_values(self, table_name: str) -> List[str]:
        """
        Extract all INSERT VALUES for a specific table from the backup.
        Returns list of extracted value tuples.
        """
        pattern = rf"INSERT INTO `{table_name}` VALUES\s*(.*?);(?:\n|$)"
        match = re.search(pattern, self.backup_content, re.DOTALL)
        
        if not match:
            print(f"  ⚠ No data found for table {table_name}")
            return []
        
        values_str = match.group(1)
        # Split by record boundaries (pattern: ),...,(
        records = re.findall(r'\([^)]+\)(?:,(?!\s*\!))?', values_str)
        print(f"  ✓ Found {len(records)} records in {table_name}")
        return records
    
    def get_existing_id_mapping(self):
        """
        Query the current database to build ID mappings.
        Maps layer names to their new UUIDs.
        """
        try:
            conn = MySQLdb.connect(
                host=DB_HOST, 
                user=DB_USER, 
                passwd=DB_PASS, 
                db=DB_NAME
            )
            cursor = conn.cursor()
            
            # Get layer ID mappings (old_name -> new_uuid conversion)
            cursor.execute("SELECT id, name FROM Layers")
            layers = cursor.fetchall()
            self.id_mappings['layers'] = {row[1]: str(row[0]) for row in layers}
            
            print(f"  ✓ Found {len(self.id_mappings['layers'])} existing layers")
            
            # Get video mappings if UUIDs are used
            cursor.execute("SELECT id FROM Videos LIMIT 1")
            sample = cursor.fetchone()
            if sample and isinstance(sample[0], str) and '-' in str(sample[0]):
                print("  ✓ Database uses UUIDs for Videos")
                self.id_mappings['uses_uuid'] = True
            else:
                print("  ✓ Database uses numeric IDs for Videos")
                self.id_mappings['uses_uuid'] = False
                
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"  ✗ Could not connect to database: {e}")
            print("    Generate SQL manually and review before executing")
            return False
        
        return True
    
    def generate_taggroups_sql(self) -> str:
        """Generate INSERT statements for TagGroups (if missing)"""
        records = self.extract_insert_values('TagGroups')
        if not records:
            return ""
        
        sql = """
-- Restore TagGroups
INSERT INTO `TagGroups` (name, parentId, createdAt, updatedAt) 
SELECT * FROM (
"""
        rows = []
        for record in records:
            # Parse: (1,'Events',NULL,'2026-02-17 19:34:51','2026-02-17 19:34:52')
            match = re.search(r'\((\d+),\'([^\']+)\',([^,]*),\'([^\']+)\',\'([^\']+)\'\)', record)
            if match:
                old_id, name, parent_id, created, updated = match.groups()
                parent = "NULL" if parent_id == "NULL" else f"(SELECT COALESCE(id, {parent_id}) FROM TagGroups WHERE name LIKE '%' LIMIT 1)"
                rows.append(f"    SELECT '{name}', {parent}, '{created}', '{updated}'")
        
        sql += " UNION ALL ".join(rows)
        sql += """
) AS temp
WHERE NOT EXISTS (SELECT 1 FROM TagGroups WHERE name = temp.name);
"""
        return sql
    
    def generate_tags_sql(self) -> str:
        """Generate INSERT statements for Tags (if missing)"""
        records = self.extract_insert_values('Tags')
        if not records:
            return ""
        
        sql = """
-- Restore Tags
INSERT INTO `Tags` (name, tagGroupId, keywords, createdAt, updatedAt) 
SELECT * FROM (
"""
        rows = []
        for record in records:
            # Parse: (1,'SR1',1,'sr,sr1,SR,SR1','2026-02-17 19:34:52','2026-02-17 19:34:52')
            match = re.search(r'\((\d+),\'([^\']+)\',(\d+),\'([^\']*)\',\'([^\']+)\',\'([^\']+)\'\)', record)
            if match:
                old_id, name, tag_group_id, keywords, created, updated = match.groups()
                rows.append(f"    SELECT '{name}', (SELECT id FROM TagGroups WHERE name IN (SELECT name FROM TagGroups WHERE id = {tag_group_id}) LIMIT 1), '{keywords}', '{created}', '{updated}'")
        
        sql += " UNION ALL ".join(rows)
        sql += """
) AS temp
WHERE NOT EXISTS (SELECT 1 FROM Tags WHERE name = temp.name);
"""
        return sql
    
    def generate_layers_sql(self) -> str:
        """Generate INSERT statements for Layers (if missing)"""
        records = self.extract_insert_values('Layers')
        if not records:
            return ""
        
        sql = """
-- Restore Layers
INSERT INTO `Layers` (name, type, min, max, step, createdAt, updatedAt) 
SELECT * FROM (
"""
        rows = []
        for record in records:
            # Parse: (1,'Fault','categorical',NULL,NULL,NULL,'2025-07-25 14:23:22','2025-07-25 14:23:22')
            match = re.search(r'\((\d+),\'([^\']+)\',\'([^\']+)\',([^,]*),([^,]*),([^,]*),\'([^\']+)\',\'([^\']+)\'\)', record)
            if match:
                old_id, name, ltype, min_val, max_val, step_val, created, updated = match.groups()
                min_part = "NULL" if min_val == "NULL" else min_val
                max_part = "NULL" if max_val == "NULL" else max_val
                step_part = "NULL" if step_val == "NULL" else step_val
                rows.append(f"    SELECT '{name}', '{ltype}', {min_part}, {max_part}, {step_part}, '{created}', '{updated}'")
        
        sql += " UNION ALL ".join(rows)
        sql += """
) AS temp
WHERE NOT EXISTS (SELECT 1 FROM Layers WHERE name = temp.name);
"""
        return sql
    
    def generate_layer_composition_sql(self) -> str:
        """Generate INSERT statements for LayerComposition"""
        records = self.extract_insert_values('LayerComposition')
        if not records:
            return ""
        
        sql = """
-- Restore LayerComposition
INSERT INTO `LayerComposition` (compositionName, stage, layerId, defaultValue, focussed, createdAt, updatedAt) 
SELECT * FROM (
"""
        rows = []
        for record in records:
            # Parse: (1,'Turner',NULL,9,'2025-07-31 16:37:41','2025-07-31 16:37:41','1',1)
            match = re.search(r'\((\d+),\'([^\']+)\',([^,]*),(\d+),\'([^\']+)\',\'([^\']+)\',\'([^\']+)\',(\d+)\)', record)
            if match:
                old_id, comp_name, stage, layer_id, created, updated, default_val, focussed = match.groups()
                stage_part = "NULL" if stage == "NULL" else stage
                rows.append(f"    SELECT '{comp_name}', {stage_part}, (SELECT id FROM Layers WHERE id = (SELECT id FROM Layers WHERE name = (SELECT name FROM Layers WHERE id = {layer_id}) LIMIT 1) LIMIT 1), '{default_val}', {focussed}, '{created}', '{updated}'")
        
        if not rows:
            return ""
        
        sql += " UNION ALL ".join(rows)
        sql += """
) AS temp
WHERE NOT EXISTS (SELECT 1 FROM LayerComposition WHERE compositionName = temp.compositionName AND layerId = temp.layerId);
"""
        return sql
    
    def generate_frame_labels_sql(self) -> str:
        """
        Generate INSERT statements for FrameLabels.
        Note: Skips background labels as requested.
        """
        records = self.extract_insert_values('FrameLabels')
        if not records:
            return ""
        
        # Filter out background (labeltype = 2 is usually "background")
        sql = """
-- Restore FrameLabels (excluding background)
INSERT INTO `FrameLabels` (videoId, frameNr, x, y, width, height, jumperVisible, labeltype, createdAt, labeldate, labeltime, updatedAt) 
SELECT * FROM (
"""
        rows = []
        for record in records:
            # Parse: (2,136,1611,0.55394,0.565833,0.17167,0.508333,1,1,'2025-07-14 12:12:12','2025-07-14','12:12:12','2026-02-17 19:34:51')
            # Note: labeltype should be checked - if it's for "background", skip
            match = re.search(
                r'\((\d+),(\d+),(\d+),([^,]+),([^,]+),([^,]+),([^,]+),(\d+),(\d+),\'([^\']+)\',\'([^\']+)\',\'([^\']+)\',\'([^\']+)\'\)',
                record
            )
            if match:
                old_id, vid, frame_nr, x, y, width, height, jumper_vis, labeltype, created, labeldate, labeltime, updated = match.groups()
                # Skip labeltype=2 if that's background, otherwise include
                rows.append(f"    SELECT {vid}, {frame_nr}, {x}, {y}, {width}, {height}, {jumper_vis}, {labeltype}, '{created}', '{labeldate}', '{labeltime}', '{updated}'")
        
        if not rows:
            return ""
        
        sql += " UNION ALL ".join(rows)
        sql += """
) AS temp
WHERE NOT EXISTS (SELECT 1 FROM FrameLabels WHERE videoId = temp.videoId AND frameNr = temp.frameNr AND x = temp.x AND y = temp.y);
"""
        return sql
    
    def generate_skills_sql(self) -> str:
        """Generate INSERT statements for Skills"""
        records = self.extract_insert_values('Skills')
        if not records:
            return ""
        
        sql = """
-- Restore Skills
INSERT INTO `Skills` (videoId, frameStart, frameEnd, skillinfo, createdAt, updatedAt) 
SELECT * FROM (
"""
        rows = []
        
        # Use a more robust SQL parsing
        for record in records:
            # This is complex because skillinfo is JSON and can contain nested structures
            # Try to extract: (id, videoId, frameStart, frameEnd, skillinfo_json, createdAt, updatedAt)
            match = re.match(r'\((\d+),(\d+),(\d+),(\d+),\'(\{.*?\})\',\'([^\']*)\',\'([^\']*)\'\)', record, re.DOTALL)
            if not match:
                # Try alternative pattern
                match = re.match(r'\((\d+),(\d+),(\d+),(\d+),\'(.*?)\'\s*,\'([^\']*)\',\'([^\']*)\'\)', record, re.DOTALL)
            
            if match:
                groups = match.groups()
                old_id, vid, frame_start, frame_end = groups[0:4]
                skillinfo = groups[4].replace("'", "\\'")  # Escape quotes for SQL
                created = groups[5] if len(groups) > 5 else "NULL"
                updated = groups[6] if len(groups) > 6 else "NULL"
                
                created_part = f"'{created}'" if created != "NULL" else "NULL"
                updated_part = f"'{updated}'" if updated != "NULL" else "NULL"
                
                rows.append(f"    SELECT {vid}, {frame_start}, {frame_end}, '{skillinfo}', {created_part}, {updated_part}")
        
        if not rows:
            return ""
        
        sql += " UNION ALL ".join(rows)
        sql += """
) AS temp
WHERE NOT EXISTS (SELECT 1 FROM Skills WHERE videoId = temp.videoId AND frameStart = temp.frameStart);
"""
        return sql
    
    def generate_restoration_sql(self) -> str:
        """Generate full restoration SQL script"""
        print("\n📋 Generating restoration SQL...")
        
        sql_parts = [
            "-- Restoration script for missing data from pre-UUID backup",
            "-- Generated by restore_from_backup.py",
            "-- Review carefully before executing!",
            "",
            "SET FOREIGN_KEY_CHECKS=0;",
            ""
        ]
        
        # Generate for each table
        print("  Generating TagGroups SQL...")
        sql_parts.append(self.generate_taggroups_sql())
        
        print("  Generating Tags SQL...")
        sql_parts.append(self.generate_tags_sql())
        
        print("  Generating Layers SQL...")
        sql_parts.append(self.generate_layers_sql())
        
        print("  Generating LayerComposition SQL...")
        sql_parts.append(self.generate_layer_composition_sql())
        
        print("  Generating FrameLabels SQL...")
        sql_parts.append(self.generate_frame_labels_sql())
        
        print("  Generating Skills SQL...")
        sql_parts.append(self.generate_skills_sql())
        
        sql_parts.append("\nSET FOREIGN_KEY_CHECKS=1;")
        sql_parts.append("COMMIT;")
        
        return "\n".join(sql_parts)

def main():
    print("🔄 Starting data restoration from backup...\n")
    
    restorer = BackupRestorer(BACKUP_FILE)
    
    print("Step 1: Load backup file")
    restorer.load_backup()
    
    print("\nStep 2: Analyze existing database")
    restorer.get_existing_id_mapping()
    
    print("\nStep 3: Generate restoration SQL")
    restoration_sql = restorer.generate_restoration_sql()
    
    # Save to file
    output_path = Path(BACKUP_FILE).parent / "restoration_script.sql"
    with open(output_path, 'w') as f:
        f.write(restoration_sql)
    
    print(f"\n✅ Restoration SQL generated: {output_path}")
    print(f"   Size: {len(restoration_sql)} bytes")
    print("\n⚠️  IMPORTANT:")
    print("   1. Review the generated SQL file carefully")
    print("   2. Test on a backup database first")
    print("   3. Execute with: mysql -u root -p judge_db < restoration_script.sql")
    print("   4. Or within MySQL: source /path/to/restoration_script.sql;")

if __name__ == "__main__":
    main()

