from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import base64
import io
import tempfile
from PIL import Image

# Try to import AI models
try:
    from transformers import pipeline
    import torch
    import cv2
    import numpy as np
    AI_AVAILABLE = True
    print("✅ AI Models Loading...")
    
    # Load NSFW (Nudity/Sexual) Detection Model
    print("📥 Loading nudity/sexual content detection model...")
    nsfw_detector = pipeline("image-classification", 
                             model="Falconsai/nsfw_image_detection")
    
    # Load Violence Detection Model
    print("📥 Loading violence/fight/kill detection model...")
    try:
        violence_detector = pipeline("image-classification",
                                     model="Nickil21/violence-detection-model")
        print("✅ Violence detection model loaded")
    except:
        try:
            violence_detector = pipeline("image-classification",
                                         model="google/vit-base-patch16-224")
            print("✅ Alternative violence model loaded")
        except:
            print("⚠️ Using NSFW model for violence detection")
            violence_detector = nsfw_detector
    
    print("✅ All AI Models Loaded Successfully!")
    
except Exception as e:
    AI_AVAILABLE = False
    print(f"⚠️ AI Models not available: {e}")
    print("⚠️ Using basic detection only")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.baseline_agent import BaselineAgent
from environment.models import ModerationAction

app = Flask(__name__)
CORS(app)
agent = BaselineAgent(use_openai=False)

# COMPREHENSIVE TEXT MODERATION KEYWORDS
TOXIC_WORDS = {
    'violence': [
        'kill', 'murder', 'death', 'die', 'dead', 'blood', 'gore', 'attack', 
        'fight', 'fighting', 'hit', 'punch', 'kick', 'stab', 'shoot', 'gun', 
        'knife', 'weapon', 'bomb', 'explode', 'torture', 'abuse', 'assault',
        'hurt', 'pain', 'injury', 'wound', 'slaughter', 'execute', 'violent',
        'aggressive', 'rage', 'destroy', 'damage', 'crash', 'collision'
    ],
    'nudity': [
        'nude', 'naked', 'nudity', 'bare', 'exposed', 'lingerie', 'bikini', 
        'swimsuit', 'underwear', 'bra', 'panties', 'thong', 'sex', 'sexual', 
        'porn', 'porno', 'xxx', 'nsfw', 'erotic', 'adult', 'explicit', 
        'provocative', 'revealing', 'short skirt', 'hot', 'sexy', 'seductive',
        'topless', 'bottomless', 'strip', 'stripper', 'intimate', 'orgy'
    ],
    'hate': [
        'hate', 'angry', 'anger', 'fury', 'rage', 'mad', 'furious', 'annoyed',
        'stupid', 'idiot', 'moron', 'retard', 'fool', 'dumb', 'worthless',
        'trash', 'garbage', 'pathetic', 'loser', 'useless', 'disgusting',
        'fuck', 'shit', 'damn', 'bitch', 'asshole', 'bastard', 'crap', 'piss',
        'whore', 'slut', 'dick', 'cock', 'cunt', 'motherfucker'
    ],
    'clothing': [
        'short', 'mini', 'tiny', 'small', 'tight', 'see-through', 'transparent',
        'revealing', 'low cut', 'deep neck', 'backless', 'strapless', 'spaghetti',
        'crop top', 'hot pants', 'short shorts', 'bikini', 'swimwear'
    ]
}

ALL_BAD_WORDS = (TOXIC_WORDS['violence'] + TOXIC_WORDS['nudity'] + 
                 TOXIC_WORDS['hate'] + TOXIC_WORDS['clothing'])

def moderate_text(text):
    text_lower = text.lower()
    violations = []
    harmful_score = 0.0
    
    for word in TOXIC_WORDS['violence']:
        if word in text_lower:
            violations.append('violence')
            harmful_score += 0.4
            break
    
    for word in TOXIC_WORDS['nudity']:
        if word in text_lower:
            violations.append('nudity')
            harmful_score += 0.4
            break
    
    for word in TOXIC_WORDS['hate']:
        if word in text_lower:
            violations.append('hate_speech')
            harmful_score += 0.35
            break
    
    if 'nudity' not in violations:
        for word in TOXIC_WORDS['clothing']:
            if word in text_lower:
                violations.append('inappropriate_clothing')
                harmful_score += 0.3
                break
    
    harmful_score = min(1.0, harmful_score)
    violations = list(set(violations))
    return violations, harmful_score

def analyze_image_with_ai(image_data):
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        violations = []
        harmful_score = 0.0
        detection_details = []
        
        if AI_AVAILABLE:
            try:
                nsfw_result = nsfw_detector(image)
                is_nsfw = nsfw_result[0]['label'] == 'nsfw'
                nsfw_confidence = nsfw_result[0]['score'] if is_nsfw else 0
                
                if is_nsfw:
                    violations.append('nudity')
                    harmful_score = max(harmful_score, nsfw_confidence)
                    detection_details.append(f"Nudity: {nsfw_confidence:.0%}")
                    print(f"   ⚠️ NUDITY detected ({nsfw_confidence:.0%})")
            except Exception as e:
                print(f"   NSFW check error: {e}")
            
            try:
                violence_result = violence_detector(image)
                is_violent = False
                violence_confidence = 0.0
                
                violence_keywords = ['violence', 'blood', 'gore', 'fight', 'attack', 
                                    'weapon', 'gun', 'knife', 'injury', 'wound',
                                    'violent', 'fighting', 'killing', 'murder', 'death']
                
                for result in violence_result:
                    label = result['label'].lower()
                    score = result['score']
                    
                    if any(keyword in label for keyword in violence_keywords):
                        is_violent = True
                        violence_confidence = max(violence_confidence, score)
                
                if is_violent:
                    violations.append('violence')
                    harmful_score = max(harmful_score, violence_confidence)
                    detection_details.append(f"Violence: {violence_confidence:.0%}")
                    print(f"   ⚠️ VIOLENCE detected ({violence_confidence:.0%})")
            except Exception as e:
                print(f"   Violence check error: {e}")
        
        if 'nudity' not in violations and harmful_score < 0.5:
            try:
                rgb_image = image.convert('RGB')
                pixels = list(rgb_image.getdata())
                skin_colors = 0
                for pixel in pixels:
                    r, g, b = pixel
                    if r > 100 and g > 50 and b > 30 and abs(r-g) < 50:
                        skin_colors += 1
                
                skin_ratio = skin_colors / len(pixels)
                if skin_ratio > 0.3:
                    violations.append('inappropriate_clothing')
                    harmful_score = max(harmful_score, 0.4)
                    detection_details.append(f"Exposed skin: {skin_ratio:.0%}")
            except:
                pass
        
        harmful_score = min(1.0, harmful_score)
        violations = list(set(violations))
        
        if detection_details:
            print(f"   📊 Detection: {', '.join(detection_details)}")
        
        return violations, harmful_score
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return [], 0.0

def analyze_video_with_ai(video_data):
    try:
        if ',' in video_data:
            video_data = video_data.split(',')[1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(base64.b64decode(video_data))
            video_path = tmp_file.name
        
        print(f"   📹 Analyzing video frame by frame...")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"   🎬 Video: {frame_count} frames, {duration:.1f} seconds")
        
        violations = []
        harmful_score = 0.0
        frames_checked = 0
        detection_details = []
        
        sample_rate = max(1, frame_count // 15)
        max_frames = min(50, frame_count)
        
        for i in range(0, min(frame_count, max_frames * sample_rate), sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            if AI_AVAILABLE:
                try:
                    nsfw_result = nsfw_detector(pil_image)
                    is_nsfw = nsfw_result[0]['label'] == 'nsfw'
                    nsfw_confidence = nsfw_result[0]['score'] if is_nsfw else 0
                    
                    if is_nsfw:
                        violations.append('nudity')
                        harmful_score = max(harmful_score, nsfw_confidence)
                        detection_details.append(f"Nudity at frame {i}: {nsfw_confidence:.0%}")
                        print(f"   ⚠️ NUDITY detected at frame {i} ({nsfw_confidence:.0%})")
                        break
                except:
                    pass
                
                try:
                    violence_result = violence_detector(pil_image)
                    is_violent = False
                    violence_confidence = 0.0
                    
                    violence_keywords = ['violence', 'blood', 'gore', 'fight', 'attack', 
                                        'weapon', 'gun', 'knife', 'injury', 'wound',
                                        'violent', 'fighting', 'killing', 'murder', 
                                        'death', 'kill', 'stab', 'shoot', 'punch', 'hit']
                    
                    for result in violence_result:
                        label = result['label'].lower()
                        score = result['score']
                        
                        if any(keyword in label for keyword in violence_keywords):
                            is_violent = True
                            violence_confidence = max(violence_confidence, score)
                    
                    if is_violent:
                        violations.append('violence')
                        harmful_score = max(harmful_score, violence_confidence)
                        detection_details.append(f"Violence at frame {i}: {violence_confidence:.0%}")
                        print(f"   ⚠️ VIOLENCE detected at frame {i} ({violence_confidence:.0%})")
                        break
                except:
                    pass
                
                try:
                    if 'anger' not in violations:
                        frame_array = np.array(frame_rgb)
                        avg_red = np.mean(frame_array[:,:,0])
                        avg_green = np.mean(frame_array[:,:,1])
                        if avg_red > avg_green * 1.3:
                            violations.append('anger')
                            harmful_score = max(harmful_score, 0.5)
                            detection_details.append(f"Anger detected at frame {i}")
                            print(f"   ⚠️ ANGER detected at frame {i}")
                except:
                    pass
            
            frames_checked += 1
        
        cap.release()
        os.unlink(video_path)
        
        harmful_score = min(1.0, harmful_score)
        violations = list(set(violations))
        
        print(f"   📊 Checked {frames_checked} frames")
        if detection_details:
            print(f"   📝 Details: {', '.join(detection_details)}")
        
        return violations, harmful_score
        
    except Exception as e:
        print(f"   ❌ Error analyzing video: {e}")
        return [], 0.0

def analyze_video_basic(video_info):
    filename = video_info.get('filename', '').lower()
    size = video_info.get('size', 0)
    
    violations = []
    harmful_score = 0.0
    
    for word in ALL_BAD_WORDS:
        if word in filename:
            violations.append('suspicious_filename')
            harmful_score += 0.3
            break
    
    if size > 100 * 1024 * 1024:
        violations.append('file_too_large')
        harmful_score += 0.3
    
    harmful_score = min(1.0, harmful_score)
    return list(set(violations)), harmful_score

@app.route('/moderate', methods=['POST'])
def moderate():
    data = request.json
    content = data.get('content', {})
    content_type = content.get('type', 'text')
    
    violations = []
    harmful_score = 0.0
    detection_method = "basic"
    
    print(f"\n{'='*50}")
    print(f"🔍 Analyzing {content_type.upper()}...")
    print(f"{'='*50}")
    
    if content_type == 'text':
        text = content.get('data', '')
        violations, harmful_score = moderate_text(text)
        detection_method = "text_keywords"
        print(f"   Text: {text[:100]}...")
        
    elif content_type == 'image':
        image_data = content.get('data', '')
        if AI_AVAILABLE and image_data:
            violations, harmful_score = analyze_image_with_ai(image_data)
            detection_method = "ai_image_model"
            print(f"   Image analysis complete - AI used")
        else:
            violations = []
            harmful_score = 0.1
            detection_method = "basic"
            print(f"   Image analysis - AI not available")
            
    elif content_type == 'video':
        video_data = content.get('data', '')
        video_info = content
        
        if AI_AVAILABLE and video_data:
            violations, harmful_score = analyze_video_with_ai(video_data)
            detection_method = "ai_video_model (frame analysis)"
            print(f"   Video analysis complete - AI frame analysis")
        else:
            violations, harmful_score = analyze_video_basic(video_info)
            detection_method = "basic_metadata"
            print(f"   Video analysis - basic only (filename check)")
    
    print(f"   📊 Harmful Score: {harmful_score:.2f}")
    print(f"   🚨 Violations: {violations if violations else 'None'}")
    print(f"   🔧 Method: {detection_method}")
    
    if harmful_score > 0.6:
        action = 'REMOVE'
        action_message = f"❌ Cannot upload - {detection_method} detected harmful content ({int(harmful_score*100)}%)"
    elif harmful_score > 0.3:
        action = 'FLAG'
        action_message = f"⚠️ Content flagged for review - {detection_method} found issues ({int(harmful_score*100)}%)"
    else:
        action = 'ALLOW'
        action_message = f"✅ Content approved - {detection_method} says safe ({int((1-harmful_score)*100)}%)"
    
    if 'nudity' in violations:
        action_message = "❌ Cannot upload - Nudity/Sexual content detected"
    elif 'violence' in violations:
        action_message = "❌ Cannot upload - Violence/Fight/Kill content detected"
    elif 'hate_speech' in violations:
        action_message = "❌ Cannot upload - Hate speech/Anger content detected"
    elif 'inappropriate_clothing' in violations:
        action_message = "❌ Cannot upload - Inappropriate clothing/Revealing content detected"
    elif 'anger' in violations:
        action_message = "❌ Cannot upload - Anger/Aggressive behavior detected"
    elif 'suspicious_filename' in violations:
        action_message = "⚠️ Content flagged - Suspicious filename detected"
    
    print(f"   🎯 Action: {action}")
    print(f"   💬 Message: {action_message}")
    print(f"{'='*50}\n")
    
    return jsonify({
        'action': action,
        'harmful_score': harmful_score,
        'confidence': harmful_score,
        'violations': violations,
        'reason': action_message,
        'content_type': content_type,
        'detection_method': detection_method,
        'ai_available': AI_AVAILABLE
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'ai_available': AI_AVAILABLE,
        'capabilities': {
            'text': True,
            'image': AI_AVAILABLE,
            'video': AI_AVAILABLE
        }
    })

@app.route('/reset', methods=['POST'])
def reset():
    """OpenEnv required endpoint - resets the environment state"""
    # Reset any global state if needed
    # For now, just return success
    return jsonify({
        'status': 'ok',
        'message': 'Environment reset successfully'
    }), 200

@app.route('/')
def home():
    return jsonify({
        "message": "AI Content Moderation API is running",
        "endpoints": {
            "health": "/health",
            "reset": "/reset (POST)",
            "moderate": "/moderate (POST)"
        },
        "status": "active",
        "ai_available": AI_AVAILABLE
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 7860))
    print("=" * 60)
    print("🚀 AI Content Moderation API Server")
    print("=" * 60)
    print(f"📡 Running on: http://0.0.0.0:{port}")
    print("📝 Endpoints:")
    print("   GET  /health - Health check")
    print("   POST /reset - Reset environment")
    print("   POST /moderate - Moderate content")
    print("   GET  / - API info")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=port)
