# scripts/cloudinary_uploader.py
import os
import yaml

# Check if cloudinary is installed
try:
    import cloudinary
    import cloudinary.uploader
    CLOUDINARY_AVAILABLE = True
except ImportError:
    CLOUDINARY_AVAILABLE = False

def get_cloudinary_config(config_path='pipeline-config.yaml'):
    """Loads Cloudinary configuration from pipeline-config.yaml."""
    if not os.path.exists(config_path):
        config_path = os.path.join('..', config_path)
    if not os.path.exists(config_path):
        return None
        
    try:
        with open(config_path, 'r') as file:
            cfg = yaml.safe_load(file)
            return cfg.get('cloudinary', None)
    except Exception:
        return None

def upload_image_to_cloud(local_image_path, public_id=None):
    """
    Uploads an image to Cloudinary if credentials are configured.
    Falls back to copying/retaining local image path if Cloudinary is not configured.
    """
    if not os.path.exists(local_image_path):
        print(f"[ERROR] Image path does not exist: {local_image_path}")
        return None

    # Load configuration
    cloud_cfg = get_cloudinary_config()
    
    # Check if credentials are set
    has_credentials = False
    if cloud_cfg:
        cloud_name = cloud_cfg.get('cloud_name', '')
        api_key = cloud_cfg.get('api_key', '')
        api_secret = cloud_cfg.get('api_secret', '')
        if cloud_name and api_key and api_secret:
            has_credentials = True
            
    if CLOUDINARY_AVAILABLE and has_credentials:
        try:
            # Configure Cloudinary
            cloudinary.config(
                cloud_name=cloud_cfg['cloud_name'],
                api_key=str(cloud_cfg['api_key']),
                api_secret=cloud_cfg['api_secret'],
                secure=True
            )
            
            # Upload image
            print(f"Uploading {local_image_path} to Cloudinary...")
            response = cloudinary.uploader.upload(
                local_image_path,
                public_id=public_id,
                folder="ritu_darpan_twin"
            )
            
            secure_url = response.get('secure_url')
            print(f"[SUCCESS] Cloudinary Upload Complete. URL: {secure_url}")
            return secure_url
        except Exception as e:
            print(f"[WARNING] Cloudinary upload failed: {e}. Falling back to local file.")
            
    else:
        if not CLOUDINARY_AVAILABLE:
            print("[INFO] Cloudinary SDK not available. Using local path.")
        else:
            print("[INFO] Cloudinary credentials missing in config. Using local fallback.")

    # Local fallback: Make sure the image is in a local directory that the dashboard can serve
    # For Streamlit, we can just return the local absolute or relative path.
    return local_image_path
