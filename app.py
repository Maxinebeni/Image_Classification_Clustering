import streamlit as st
import os
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def main():
    st.title("Clustering of Mens Shoes similarity")
    st.sidebar.header("Add your Shoe Data")

    # Upload image dataset
    uploaded_file = st.sidebar.file_uploader("Upload zip file containing images", type=["zip"])

    if uploaded_file is not None:
        st.sidebar.markdown("### Sample Images")
        sample_images = extract_images_from_zip(uploaded_file)
        for image in sample_images:
            st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)

        # Get the current directory
        current_dir = os.path.dirname(os.path.realpath(__file__))

        # Determine number of clusters based on unique categories
        num_clusters = len(set([image.split("_")[0] for image in os.listdir(os.path.join(current_dir, "archive (8)"))]))

        if st.sidebar.button("Cluster Images"):
            st.sidebar.text("Clustering in progress...")

            # K-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(np.array(sample_images).reshape(len(sample_images), -1))

            # Get cluster labels
            cluster_labels = kmeans.labels_

            # Display clustering results
            clustered_images = {}
            for i in range(num_clusters):
                clustered_images[i] = []

            for idx, label in enumerate(cluster_labels):
                clustered_images[label].append((f"Image {idx + 1}", sample_images[idx]))

            st.success("Clustering completed!")
            st.write("Clustered Images:")
            for cluster, images in clustered_images.items():
                st.write(f"Cluster {cluster + 1}: {len(images)} images")
                for i, (image_name, image) in enumerate(images[:3]):  # Display only the first 3 images
                    st.image(image, caption=image_name, use_column_width=True)
                    if i == 2 and len(images) > 3:
                        st.write(f"... and {len(images) - 3} more images")

def extract_images_from_zip(uploaded_zip):
    import zipfile
    import io

    sample_images = []
    with zipfile.ZipFile(uploaded_zip) as z:
        for filename in z.namelist():
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                with z.open(filename) as f:
                    img = Image.open(io.BytesIO(f.read()))
                    sample_images.append(np.array(img))
    return sample_images

if __name__ == "__main__":
    main()
