import csv
import json
from pathlib import Path

videos_dir = Path('data/real_videos_canonical')
manifest = []

for video_file in sorted(videos_dir.glob('*.mp4')):
    gloss = video_file.stem
    stat = video_file.stat()
    manifest.append({
        'gloss': gloss,
        'video_file': video_file.name,
        'relative_path': f'real_videos_canonical/{video_file.name}',
        'size_bytes': stat.st_size,
        'size_mb': round(stat.st_size / (1024*1024), 2),
        'source': 'ISLRTC_dictionary (silentone0725 re-encoded)',
        'license': 'MIT (original: Government open data / check data.gov.in)',
    )

manifest_path = Path('data/real_videos_canonical/manifest.csv')
with open(manifest_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
    writer.writeheader()
    writer.writerows(manifest)

print(f'Manifest written to {manifest_path} with {len(manifest)} entries')

json_path = Path('data/real_videos_canonical/manifest.json')
with open(json_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f'JSON manifest written to {json_path}')