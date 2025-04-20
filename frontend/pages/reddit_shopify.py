import streamlit as st
import requests
import base64
import json
from io import BytesIO  # Add this import
from requests.exceptions import RequestException
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
 
st.set_page_config(page_title="REDDIT SHPOIFY", layout="centered")
st.title("🧠 Grok AI Product Poster Generator")
st.markdown("Generate photorealistic e-commerce product posters using **Grok (xAI)**.")
 
@st.cache_resource
def get_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session
 
# Add before the form
def check_server():
    try:
        response = requests.get("http://34.172.249.210:8016/health", timeout=5)
        return response.ok
    except:
        return False
 
# === Form Inputs === #
with st.form("generate_form"):
    title = st.text_input("🛍️ Product Title", placeholder="Enter product name", max_chars=100)
 
    # Add after title
    if not check_server():
        st.error("⚠️ MCP Server is not running. Please start the server first.")
        st.stop()
 
    description = st.text_area("📝 Product Description", placeholder="Write a compelling product description")
    submitted = st.form_submit_button("🚀 Generate Poster")
 
# === POST to MCP Server === #
if submitted:
    if not title or not description:
        st.warning("Please fill in both the product title and description.")
    else:
        with st.spinner("Generating image using Grok..."):
            try:
                url = "http://34.172.249.210:8016/generate"  # Updated endpoint
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                payload = {
                    "title": title,
                    "description": description,
                }
               
                # Show request details in debug mode
                if st.checkbox("Debug Mode"):
                    st.code(f"Request URL: {url}\nPayload: {json.dumps(payload, indent=2)}")
               
                session = get_session()
                response = session.post(
                    url=url,
                    json=payload,
                    headers=headers,
                    timeout=60
                )
               
                # Check status code first
                response.raise_for_status()
               
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON response: {response.text}")
                    st.error(f"JSON Error: {str(e)}")
                    st.stop()
 
                # Update the image processing section
                if result.get("status") == "success":
                    st.success("Poster generated successfully!")
                   
                    image_url = result.get("imageUrl")
                    if image_url:
                        try:
                            # Validate image URL format
                            if image_url == "data:image/png;base64,...":
                                st.error("Received placeholder image data from server")
                                st.info("Please check server implementation - actual base64 image data required")
                                st.stop()
 
                            # Show detailed debug info
                            with st.expander("🔍 Debug Image Data"):
                                st.code(f"""
                                Image URL length: {len(image_url)}
                                First 50 chars: {image_url[:50]}
                                Is base64 prefix present: {"base64," in image_url}
                                """)
 
                            # Extract and validate base64 data
                            if "base64," in image_url:
                                base64_data = image_url.split("base64,")[1].strip()
                            else:
                                base64_data = image_url.strip()
 
                            if not base64_data or base64_data == "...":
                                raise ValueError("Invalid or empty base64 data received from server")
 
                            # Decode and process image
                            image_bytes = base64.b64decode(base64_data)
                           
                            # Show binary data debug info
                            with st.expander("🔍 Binary Data Info"):
                                PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
                                debug_info = (
                                    f"Bytes length: {len(image_bytes)}\n"
                                    f"First 16 bytes: {image_bytes[:16].hex()}\n"
                                    f"PNG signature match: {image_bytes.startswith(PNG_SIGNATURE)}"
                                )
                                st.code(debug_info)
 
                            # Create image buffer and verify format
                            buf = BytesIO(image_bytes)
                            try:
                                with Image.open(buf) as img:
                                    # Show image details
                                    st.info(f"Image format: {img.format}, Size: {img.size}, Mode: {img.mode}")
                                   
                                    # Convert if needed
                                    if img.mode in ('RGBA', 'LA'):
                                        img = img.convert('RGB')
                                   
                                    # Display image
                                    st.image(img, caption=title, use_column_width=True)
                                   
                                    # Reset buffer and add download
                                    buf.seek(0)
                                    st.download_button(
                                        label="📥 Download Poster",
                                        data=buf.getvalue(),
                                        file_name=f"{title.lower().replace(' ', '_')}_poster.png",
                                        mime="image/png"
                                    )
                            except Exception as img_error:
                                st.error(f"Image format error: {str(img_error)}")
                                st.warning("Make sure the server is returning a valid PNG image in base64 format")
                               
                        except Exception as e:
                            st.error(f"Image processing failed: {str(e)}")
                            if st.checkbox("Show Technical Details"):
                                st.code(f"""
                                Error Type: {type(e).__name__}
                                Error Details: {str(e)}
                                """)
                    else:
                        st.error("No image URL in server response")
                       
                    # Add after displaying the image and download button
                    if result.get("redditUrl"):
                        st.success("✅ Posted to Reddit!")
                        st.markdown(f"""
                        🔗 **Reddit Post**: [View on Reddit]({result["redditUrl"]})
                        """)
 
                    if result.get("shopifyUrl"):
                        st.success("✅ Added to Shopify Store!")
                        st.markdown(f"""
                        🛍️ **Product Page**: [View in Store]({result["shopifyUrl"]})
                        """)
 
                    # Add social sharing buttons
                    st.markdown("### 📢 Share")
                    col1, col2 = st.columns(2)
                    with col1:
                        if result.get("redditUrl"):
                            st.link_button(
                                "Share on Reddit",
                                result["redditUrl"],
                                help="View and share the Reddit post"
                            )
                    with col2:
                        if result.get("shopifyUrl"):
                            st.link_button(
                                "View in Store",
                                result["shopifyUrl"],
                                help="View the product in Shopify store"
                            )
 
                    # Move the metadata expander after the sharing buttons
                    with st.expander("📊 Generation Details"):
                        if result.get("redditUrl"):
                            st.text("Reddit Post URL:")
                            st.code(result["redditUrl"])
                        if result.get("shopifyUrl"):
                            st.text("Shopify Product URL:")
                            st.code(result["shopifyUrl"])
                        st.json({k: v for k, v in result.items() if v is not None and k not in ["redditUrl", "shopifyUrl"]})
                else:
                    st.error(f"Error: {result.get('message', 'Unknown error')}")
                    st.json(result)  # Show full error response
 
            except RequestException as e:
                st.error(f"⚠️ Network Error: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    st.error(f"Server Response: {e.response.text}")
            except Exception as e:
                st.error(f"⚠️ Unexpected Error: {str(e)}")