import os
import subprocess
import pickle
import joblib
import glob
import shutil

def run_chunk(video_source, output_dir, start, end, seq_name):
    # Use a specific output dir for this chunk to avoid collisions
    chunk_out = os.path.join(output_dir, seq_name)
    os.makedirs(chunk_out, exist_ok=True)
    
    cmd = [
        "conda", "run", "-n", "4dhumans", "python", "track.py",
        f'video.source="{video_source}"',
        f'video.output_dir="{chunk_out}"',
        "phalp.detector=maskrcnn",
        "device=cuda",
        "detect_shots=False",
        "render.type=HUMAN_BBOX",
        f"phalp.start_frame={start}",
        f"phalp.end_frame={end}",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "detectron2_repo")))

def stitch_results(output_dir, video_name, chunks):
    final_data = {}
    video_list = []
    
    # The video name produced by PHALP is based on the source filename
    base_name = os.path.basename(video_name)
    
    for i, (start, end) in enumerate(chunks):
        seq_name = f"chunk_{i}"
        chunk_out = os.path.join(output_dir, seq_name)
        
        # PHALP saves results in {chunk_out}/results/demo_{base_name}.pkl
        pkl_path = os.path.join(chunk_out, "results", f"demo_{base_name}.pkl")
        video_path = os.path.join(chunk_out, f"PHALP_{base_name}.mp4")
        
        if os.path.exists(pkl_path):
            try:
                data = joblib.load(pkl_path)
                final_data.update(data)
            except Exception as e:
                print(f"Error loading {pkl_path}: {e}")
        else:
            print(f"Warning: {pkl_path} not found")
            
        if os.path.exists(video_path):
            video_list.append(video_path)
        else:
            print(f"Warning: {video_path} not found")
    
    # Save stitched pkl
    stitched_pkl = os.path.join(output_dir, f"stitched_{base_name}.pkl")
    joblib.dump(final_data, stitched_pkl)
    print(f"Stitched result saved to {stitched_pkl}")
    
    # Stitch videos using ffmpeg
    if video_list:
        list_file = os.path.join(output_dir, "video_list.txt")
        with open(list_file, "w") as f:
            for v in video_list:
                f.write(f"file '{os.path.abspath(v)}'\n")
        
        output_video = os.path.join(output_dir, f"stitched_{base_name}.mp4")
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", list_file, "-c", "copy", output_video
        ]
        print(f"Stitching videos: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd)
        print(f"Stitched video saved to {output_video}")

if __name__ == "__main__":
    print("Starting chunked processing...")
    # Configuration
    chunk_size = 300
    
    videos = [
        ("example_data/videos/Me1.mp4", 1852, "Me1"),
        ("example_data/videos/Me2.mp4", 2741, "Me2")
    ]
    
    for video_path, total_frames, video_name in videos:
        print(f"Processing video: {video_name}")
        output_dir = f"outputs_{video_name}_chunked"
        os.makedirs(output_dir, exist_ok=True)
        
        chunks = []
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            chunks.append((start, end))
        
        for i, (s, e) in enumerate(chunks):
            seq_name = f"chunk_{i}"
            chunk_out = os.path.join(output_dir, seq_name)
            # Check if pkl already exists in the chunk dir
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            pkl_path = os.path.join(chunk_out, "results", f"demo_{base_name}.pkl")
            
            if os.path.exists(pkl_path):
                print(f"Skipping chunk {i}: frames {s} to {e} (already exists)")
                continue
            
            print(f"Running chunk {i}: frames {s} to {e}")
            run_chunk(video_path, output_dir, s, e, seq_name)
            print(f"Finished chunk {i} for {video_name}")
        
        print(f"Stitching results for {video_name}...")
        stitch_results(output_dir, base_name, chunks)
        
    print("Done with all videos.")
