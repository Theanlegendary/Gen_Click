import os
from moviepy import ImageClip, concatenate_videoclips

def merge_images_to_video(image_paths, output_filename="final_story.mp4", duration_per_image=3):
    """
    Takes a list of image paths and merges them into a single video slideshow.
    """
    if not image_paths:
        print("No images to merge.")
        return None
        
    print(f"Preparing to merge {len(image_paths)} images into a video...")
    
    clips = []
    try:
        # Load all images and set their duration (e.g., 3 seconds each)
        for path in image_paths:
            if os.path.exists(path):
                clip = ImageClip(path).with_duration(duration_per_image)
                clips.append(clip)
            else:
                print(f"Warning: File {path} not found. Skipping.")
                
        if not clips:
            print("No valid images found to merge.")
            return None
            
        # Concatenate them
        print("Stitching images together into a video...")
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Write the result to an mp4 file
        final_clip.write_videofile(
            output_filename,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            logger=None # Suppress verbose output
        )
        
        # Close the clips to free memory
        for clip in clips:
            clip.close()
        final_clip.close()
        
        print(f"Merging complete! Final video saved to: {output_filename}")
        return os.path.abspath(output_filename)
        
    except Exception as e:
        print(f"An error occurred during merging: {e}")
        return None
