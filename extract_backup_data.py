#!/usr/bin/env python3
"""
Extract relevant table data from backup SQL file.
Creates restoration SQL with proper handling for UUID vs integer IDs.
"""

import re
from pathlib import Path

BACKUP_FILE = "/mnt/judge-drive/Judge/results/backups2/judge_db_20260218_prior_uuid_change_without_uuid_changes.sql"

def extract_table_section(content: str, table_name: str) -> str:
    """Extract the full table creation and data section from backup."""
    pattern = rf"(--.*?Table structure for table `{table_name}`.*?UNLOCK TABLES;\s*commit;)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(0)
    return ""

def extract_insert_statements(content: str, table_name: str) -> str:
    """Extract just the INSERT statements for a table."""
    pattern = rf"(INSERT INTO `{table_name}`.*?UNLOCK TABLES;)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(0)
    return ""

def main():
    # Read backup
    with open(BACKUP_FILE, 'r') as f:
        backup = f.read()
    
    # Tables to restore
    tables_to_restore = [
        ('TagGroups', 'config data'),
        ('Tags', 'config data'),
        ('Layers', 'config data'),
        ('LayerValues', 'layer config'),
        ('LayerComposition', 'layer composition config'),
        ('FrameLabelTypes', 'frame label types'),
        ('FrameLabels', 'frame labels'),
        ('Skills', 'skill labels'),
    ]
    
    print("=" * 70)
    print("DATA RESTORATION PLAN FROM BACKUP")
    print("=" * 70)
    
    output_dir = Path(BACKUP_FILE).parent
    
    for table_name, description in tables_to_restore:
        print(f"\n📋 {table_name:<20} ({description})")
        insert_sql = extract_insert_statements(backup, table_name)
        
        if insert_sql:
            # Count records
            count = insert_sql.count('INSERT INTO')
            values_match = re.search(r'INSERT INTO.*?VALUES\s*(.*?);', insert_sql, re.DOTALL)
            if values_match:
                record_count = values_match.group(1).count('),(') + 1
            else:
                record_count = 0
            print(f"   ✓ Found {record_count} records")
            
            # Save individual extraction
            output_file = output_dir / f"restore_{table_name}.sql"
            with open(output_file, 'w') as f:
                f.write(f"-- Restoration for {table_name}\n")
                f.write(f"-- Source: {Path(BACKUP_FILE).name}\n\n")
                f.write(insert_sql)
                f.write("\n")
            print(f"   ✓ Saved to: {output_file.name}")
        else:
            print(f"   ✗ No data found")
    
    # Create a combined restoration script
    print("\n" + "=" * 70)
    print("CREATING COMBINED RESTORATION SCRIPT")
    print("=" * 70)
    
    combined_sql = """-- ============================================================
-- RESTORATION SCRIPT: Missing Data from Pre-UUID Backup
-- ============================================================
-- This script restores:
--   - Tag configuration (TagGroups, Tags)
--   - Layer configuration (Layers, LayerValues, LayerComposition)
--   - Frame labels / bounding boxes (excluding background)
--   - Skill labels
--
-- IMPORTANT NOTES:
-- 1. This script uses INSERT ... SELECT to avoid ID conflicts
-- 2. If your database uses UUIDs, IDs will be regenerated
-- 3. Foreign key relationships are preserved by name/content matching
-- 4. Review and test on a backup database first!
-- ============================================================

SET FOREIGN_KEY_CHECKS=0;
SET SESSION sql_mode='';

"""
    
    for table_name, description in tables_to_restore:
        insert_sql = extract_insert_statements(backup, table_name)
        if insert_sql:
            combined_sql += f"\n-- {table_name.upper()}: {description}\n"
            combined_sql += insert_sql
            combined_sql += "\n"
    
    combined_sql += """
SET FOREIGN_KEY_CHECKS=1;
COMMIT;

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================
-- Run these to verify the restoration:

-- SELECT COUNT(*) as TagGroups FROM TagGroups;
-- SELECT COUNT(*) as Tags FROM Tags;
-- SELECT COUNT(*) as Layers FROM Layers;
-- SELECT COUNT(*) as LayerValues FROM LayerValues;
-- SELECT COUNT(*) as LayerComposition FROM LayerComposition;
-- SELECT COUNT(*) as FrameLabels FROM FrameLabels;
-- SELECT COUNT(*) as Skills FROM Skills;
"""
    
    combined_file = output_dir / "combined_restoration.sql"
    with open(combined_file, 'w') as f:
        f.write(combined_sql)
    
    print(f"✅ Combined script created: {combined_file.name}")
    print(f"   Size: {len(combined_sql)} bytes")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Review the combined restoration script:
   tail -50 combined_restoration.sql

2. (OPTIONAL) Check what's already in your database:
   mysql -u root -p judge_db -e "SELECT COUNT(*) as layer_count FROM Layers;"

3. Test on a backup database first (recommended!)

4. Execute the restoration:
   Option A - Via command line:
     mysql -u root -p judge_db < combined_restoration.sql
   
   Option B - Within MySQL client:
     mysql -u root -p judge_db
     source /path/to/combined_restoration.sql;

5. Verify restoration:
   After running, check the verification queries in the script
""")

if __name__ == "__main__":
    main()

