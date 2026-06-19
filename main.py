import os
import sys
import time
from prompt_enhancer import generate_story_scenes
from image_generator import generate_image, download_image
from video_merger import merge_images_to_video

def main(auto_topic=None, auto_clips=None):
    print("==================================================")
    print("      100% FREE AI Storybook Video Generator      ")
    print("==================================================")
    print("Type 'exit' or 'quit' to stop the script.\n")
    
    story_counter = 1
    
    while True:
        try:
            if auto_topic:
                user_topic = auto_topic
                print(f"\n[Story {story_counter}] Auto-generating story for: {user_topic}")
            else:
                # 1. Prompt the user for their story topic
                user_topic = input(f"\n[Story {story_counter}] What is the topic of your story? > ")
                
            if user_topic.strip().lower() in ['exit', 'quit']:
                print("Exiting generator. Goodbye!")
                break
                
            if not user_topic.strip():
                print("Please enter a valid topic.")
                continue
                
            if auto_clips:
                num_clips = auto_clips
            else:
                # 2. Ask for the number of clips
                clips_input = input("How many scenes do you want for this story? (e.g., 5) > ")
                try:
                    num_clips = int(clips_input)
                    if num_clips <= 0:
                        raise ValueError
                except:
                    print("Please enter a valid positive number.")
                    continue

            print("\n1. Director AI (Gemini) is writing the storyboard scenes...")
            scenes = generate_story_scenes(user_topic, num_clips)
            
            if not scenes:
                print("Failed to generate scenes. Skipping this story.")
                continue
                
            print(f"   Successfully generated {len(scenes)} scenes!")
            
            # Create a dedicated folder for this story
            folder_name = f"story_{story_counter}_{user_topic.replace(' ', '_')[:20]}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                
            print(f"\n2. Generating Images Sequentially (Saved in folder: {folder_name})")
            
            def download_scene_image(scene_prompt, idx):
                print(f"\n--- Scene {idx+1}/{len(scenes)} ---")
                print(f"Prompt: {scene_prompt}")
                try:
                    image_url = generate_image(scene_prompt, idx+1)
                    filename = os.path.join(folder_name, f"scene_{idx+1}.jpg")
                    filepath = download_image(image_url, filename)
                    return filepath
                except Exception as e:
                    print(f"   [X] Failed to generate Scene {idx+1}: {e}")
                    from image_generator import generate_gradient_fallback
                    try:
                        filename = os.path.join(folder_name, f"scene_{idx+1}.jpg")
                        filepath = generate_gradient_fallback(scene_prompt, filename)
                        return filepath
                    except Exception as fallback_err:
                        print(f"   [X] Fallback also failed for Scene {idx+1}: {fallback_err}")
                        return None

            generated_paths = []
            for i, scene_prompt in enumerate(scenes):
                if i > 0:
                    time.sleep(0.5)
                filepath = download_scene_image(scene_prompt, i)
                if filepath:
                    generated_paths.append(filepath)
            
            if not generated_paths:
                print("\nNo images were successfully generated. Skipping merge.")
                if auto_topic: break
                continue
                
            print("\n3. Merging all successful images into a Video Slideshow...")
            final_story_path = os.path.join(folder_name, "FINAL_STORYBOOK.mp4")
            
            # Each image will show for 4 seconds in the video
            merged_video = merge_images_to_video(generated_paths, final_story_path, duration_per_image=4)
            
            if merged_video:
                print(f"\n[SUCCESS] Your complete storybook video is ready at: {merged_video}")
            else:
                print("\n[X] Failed to merge the images into a video.")
                
            story_counter += 1
            
            # If we were auto-running a specific topic, exit after finishing
            if auto_topic:
                break
            
        except KeyboardInterrupt:
            print("\nExiting generator. Goodbye!")
            break
        except Exception as e:
            print(f"\n[X] An error occurred: {e}")
            if auto_topic: break

if __name__ == "__main__":
    # If arguments are passed, run it automatically
    if len(sys.argv) > 2:
        main(auto_topic=sys.argv[1], auto_clips=int(sys.argv[2]))
    else:
        main()
