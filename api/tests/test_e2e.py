#!/usr/bin/env python3
"""
End-to-End Test Script for AI Judge API

This script tests the test API service (api-e2e-service) by simulating a complete account workflow
and populating the test database via API endpoints. It ensures e2e tests cannot run
against production by checking for test mode indicators.

Tested workflow:
1. Health check (verifies test environment)
2. Scan for videos (account provides test videos in file system)
3. Test folder/video retrieval endpoints
4. Add frame labels via API
5. Test skill/labeling endpoints
6. Launch and monitor training jobs
7. Test stats and other endpoints
"""

import requests
import time
import os
import sys
import json

API_LOCAL_PORT = os.getenv("API_LOCAL_PORT", "5555")
API_BASE_URL = f"http://api-e2e-service:{API_LOCAL_PORT}"
STORAGE_DIR_VIDEOS = os.getenv("STORAGE_DIR_VIDEOS", "/storage/videos")
STORAGE_DIR_BACKUPS = os.getenv("MYSQL_BACKUP", "/storage/backups")
STORAGE_DIR_GENERATED_DATA = os.getenv("STORAGE_DIR_GENERATED_DATA", "/storage/results")

def wait_for_api():
    """Wait for API to be ready"""
    print("⏳ Waiting for API to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    print("❌ API failed to start")
    return False

def test_health():
    """Test health endpoint"""
    print("🩺 Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

    # Ensure we're running against test API, not production
    assert data.get("test_mode") == True, "E2E tests must run against test API, not production!"
    assert data.get("database") == "test", "E2E tests must use test database!"

    print("✅ Health check passed - confirmed test environment")

def prepare_environment():
    """Create folders and make sure there are videos in it to e2e test it"""
    # Create backup dirs

    os.makedirs(STORAGE_DIR_BACKUPS, exist_ok=True)
    os.makedirs(STORAGE_DIR_VIDEOS, exist_ok=True)
    os.makedirs(STORAGE_DIR_GENERATED_DATA, exist_ok=True)

    # TODO : populate STORAGE_DIR_VIDEOS with test videos.

def scan_for_videos():
    """Scan for videos in test directory"""
    print("🎥 Scanning for videos...")
    response = requests.get(f"{API_BASE_URL}/discover")
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Discovery completed, found {len(data)} items")
    return data

def test_folder_endpoints():
    """Test all folder-related endpoints"""
    print("📁 Testing folder endpoints...")

    # Get root folders
    response = requests.get(f"{API_BASE_URL}/folders")
    assert response.status_code == 200
    root_data = response.json()
    assert "Children" in root_data
    assert "Videos" in root_data
    print(f"✅ Root folders: {len(root_data['Children'])} folders, {root_data['VideoCount']} videos")

    folders = root_data.get('Children', [])
    if folders:
        folder_id = folders[0]['Id']

        # Get specific folder
        response = requests.get(f"{API_BASE_URL}/folders/{folder_id}")
        assert response.status_code == 200
        folder_data = response.json()
        assert folder_data['Id'] == folder_id
        print(f"✅ Folder {folder_id} details retrieved")

        return folder_data
    return None

def test_video_endpoints(videos):
    """Test video-related endpoints"""
    print("🎬 Testing video endpoints...")

    if not videos:
        print("⚠️ No videos found, skipping video tests")
        return None

    video_id = list(videos.keys())[0]
    video_info = videos[video_id]

    # Get video info
    response = requests.get(f"{API_BASE_URL}/video/{video_id}/info")
    assert response.status_code == 200
    info_data = response.json()
    assert info_data['Id'] == video_id
    print(f"✅ Video {video_id} info retrieved")

    # Test video download (should return video file)
    response = requests.get(f"{API_BASE_URL}/video/{video_id}")
    assert response.status_code == 200
    assert 'video' in response.headers.get('content-type', '').lower()
    print(f"✅ Video {video_id} download works")

    # Test video image
    response = requests.get(f"{API_BASE_URL}/video/{video_id}/image")
    assert response.status_code == 200
    print(f"✅ Video {video_id} image retrieved")

    return video_id

def test_frame_labeling(video_id):
    """Test frame labeling endpoints"""
    print("🏷️ Testing frame labeling...")

    # Add a frame label
    label_data = {
        "x": 0.5,
        "y": 0.5,
        "width": 0.2,
        "height": 0.3
    }
    response = requests.post(
        f"{API_BASE_URL}/video/{video_id}/frameNr/1",
        json=label_data
    )
    assert response.status_code == 200
    print(f"✅ Frame label added to video {video_id}")

    # Test frame label types
    response = requests.get(f"{API_BASE_URL}/frameLabelTypes")
    assert response.status_code == 200
    print("✅ Frame label types retrieved")

def test_skill_endpoints(video_id):
    """Test skill-related endpoints"""
    print("⚡ Testing skill endpoints...")

    # Get skills for video
    response = requests.get(f"{API_BASE_URL}/skill/{video_id}")
    assert response.status_code == 200
    print(f"✅ Skills retrieved for video {video_id}")

    # Test skill levels
    response = requests.get(f"{API_BASE_URL}/skilllevel")
    assert response.status_code == 200
    print("✅ Skill levels retrieved")

def test_job_endpoints():
    """Test job-related endpoints"""
    print("🚀 Testing job endpoints...")

    # Get job options for localize step
    response = requests.get(f"{API_BASE_URL}/job/options/localize")
    assert response.status_code == 200
    options = response.json()
    print(f"✅ Job options retrieved: {list(options.keys()) if options else 'none'}")

    # Launch a training job
    if options:
        job_data = {
            "step": "localize",
            "model": list(options.keys())[0] if options else "yolo11n"
        }
        response = requests.post(f"{API_BASE_URL}/job", json=job_data)
        assert response.status_code in [200, 201]
        print("✅ Training job launched")
    else:
        print("⚠️ No job options available, skipping job launch")

def test_stats_endpoints():
    """Test stats endpoints"""
    print("📊 Testing stats endpoints...")

    response = requests.get(f"{API_BASE_URL}/stats")
    assert response.status_code == 200
    stats = response.json()
    print(f"✅ Stats retrieved: {list(stats.keys()) if stats else 'empty'}")

def test_tag_endpoints():
    """Test tag-related endpoints"""
    print("🏷️ Testing tag endpoints...")

    # Get tags
    response = requests.get(f"{API_BASE_URL}/tags")
    assert response.status_code == 200
    print("✅ Tags retrieved")

    # Get tag groups
    response = requests.get(f"{API_BASE_URL}/tagGroups")
    assert response.status_code == 200
    print("✅ Tag groups retrieved")

def test_ml_layer_endpoints():
    """Test ML layer endpoints"""
    print("🧠 Testing ML layer endpoints...")

    # Get layer types
    response = requests.get(f"{API_BASE_URL}/layers/types")
    assert response.status_code == 200
    print("✅ ML layer types retrieved")

    # Get layers
    response = requests.get(f"{API_BASE_URL}/layers")
    assert response.status_code == 200
    print("✅ ML layers retrieved")

def main():
    print("🧪 Starting End-to-End API Tests...")

    if not wait_for_api():
        sys.exit(1)

    try:
        test_health()

        # Creating directories, videos...
        prepare_environment()

        # Scan for account-provided test videos
        discovered = scan_for_videos()

        # Test folder endpoints
        folder_data = test_folder_endpoints()
        videos = folder_data.get('Videos', {}) if folder_data else {}

        # Test video endpoints if videos exist
        video_id = None
        if videos:
            video_id = test_video_endpoints(videos)

            # Test labeling and skills if we have a video
            if video_id:
                test_frame_labeling(video_id)
                test_skill_endpoints(video_id)

        # Test job system
        test_job_endpoints()

        # Test other endpoints
        test_stats_endpoints()
        test_tag_endpoints()
        test_ml_layer_endpoints()

        print("🎉 All E2E API tests passed!")
        sys.exit(0)

    except Exception as e:
        print(f"❌ E2E test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

