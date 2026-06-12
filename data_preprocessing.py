import os
import zipfile
import urllib.request
import numpy as np
import scipy.io as sio
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
import shutil
import h5py

# Configurations
DATASET_URL = "https://figshare.com/ndownloader/articles/1512427/versions/3"
DOWNLOAD_DIR = "dataset_raw"
ZIP_FILE_NAME = "1512427.zip"
OUTPUT_DIR = "dataset_processed"

def download_dataset():
    """Downloads the Cheng Figshare brain tumor dataset if not already present."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    zip_path = os.path.join(DOWNLOAD_DIR, ZIP_FILE_NAME)
    
    if os.path.exists(zip_path):
        print(f"[*] Zip file '{zip_path}' already exists. Skipping download.")
        return zip_path
        
    print(f"[*] Downloading Cheng Figshare dataset from: {DATASET_URL}...")
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
        urllib.request.install_opener(opener)
        
        def report_hook(block_num, block_size, total_size):
            read_so_far = block_num * block_size
            if total_size > 0:
                percent = read_so_far * 100 / total_size
                print(f"\rDownloading: {percent:.1f}% ({read_so_far / (1024*1024):.1f} MB of {total_size / (1024*1024):.1f} MB)", end="")
            else:
                print(f"\rDownloading: {read_so_far / (1024*1024):.1f} MB", end="")
        
        urllib.request.urlretrieve(DATASET_URL, zip_path, reporthook=report_hook)
        print("\n[+] Download completed successfully.")
        return zip_path
    except Exception as e:
        print(f"\n[-] Error downloading dataset: {e}")
        print("[!] Please download the zip file manually from https://doi.org/10.6084/m9.figshare.1512427")
        print(f"[!] Save it as '{ZIP_FILE_NAME}' inside the '{DOWNLOAD_DIR}' directory.")
        return None

def extract_dataset(zip_path):
    """Extracts the main zip and the nested zips containing MAT files."""
    if not zip_path or not os.path.exists(zip_path):
        print("[-] Extraction failed: zip file path is invalid.")
        return None
        
    temp_extract_dir = os.path.join(DOWNLOAD_DIR, "extracted_temp")
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    print("[*] Extracting main zip file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
        
    nested_zips = [
        "brainTumorDataPublic_1-766.zip",
        "brainTumorDataPublic_767-1532.zip",
        "brainTumorDataPublic_1533-2298.zip",
        "brainTumorDataPublic_2299-3064.zip"
    ]
    
    mat_output_dir = os.path.join(DOWNLOAD_DIR, "mat_files")
    os.makedirs(mat_output_dir, exist_ok=True)
    
    has_nested = False
    for nz in nested_zips:
        nz_path = os.path.join(temp_extract_dir, nz)
        if os.path.exists(nz_path):
            has_nested = True
            print(f"[*] Extracting nested zip: {nz}...")
            with zipfile.ZipFile(nz_path, 'r') as zip_ref:
                zip_ref.extractall(mat_output_dir)
                
    if has_nested:
        shutil.rmtree(temp_extract_dir)
        print(f"[+] All MAT files extracted to: {mat_output_dir}")
        return mat_output_dir
    else:
        # If there were no nested zips, it means the main zip contained the MAT files directly
        mat_files = []
        for root, dirs, files in os.walk(temp_extract_dir):
            for f in files:
                if f.endswith(".mat"):
                    mat_files.append(os.path.join(root, f))
        if len(mat_files) > 0:
            from collections import Counter
            folders = [os.path.dirname(f) for f in mat_files]
            mat_dir = Counter(folders).most_common(1)[0][0]
            print(f"[+] MAT files found directly in extracted zip: {mat_dir}")
            return mat_dir
            
    return None

def extract_mat_metadata(file_path):
    """Extracts ONLY label and patient ID from a MAT file to avoid reading heavy image matrices."""
    try:
        with h5py.File(file_path, 'r') as f:
            if 'cjdata' in f:
                cjdata = f['cjdata']
                label = int(np.array(cjdata['label']).flatten()[0])
                pid_arr = np.array(cjdata['PID']).flatten()
                if pid_arr.dtype.kind in ['U', 'S']:
                    pid = "".join([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in pid_arr]).strip()
                else:
                    pid = "".join(chr(int(c)) for c in pid_arr if c > 0).strip()
            else:
                label = int(np.array(f['label']).flatten()[0])
                pid_arr = np.array(f['PID']).flatten()
                if pid_arr.dtype.kind in ['U', 'S']:
                    pid = "".join([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in pid_arr]).strip()
                else:
                    pid = "".join(chr(int(c)) for c in pid_arr if c > 0).strip()
            if not pid:
                pid = "unknown"
            return label, pid
    except Exception as e_h5py:
        try:
            mat_data = sio.loadmat(file_path, variable_names=['cjdata'])
            if 'cjdata' in mat_data:
                cjdata = mat_data['cjdata'][0, 0]
            else:
                cjdata = mat_data
                
            if hasattr(cjdata, 'dtype') and 'label' in cjdata.dtype.names:
                label = int(cjdata['label'].flatten()[0])
            elif 'label' in cjdata:
                label = int(cjdata['label'].flatten()[0])
            else:
                raise KeyError("Could not find 'label' field.")
                
            if hasattr(cjdata, 'dtype') and 'PID' in cjdata.dtype.names:
                pid_arr = cjdata['PID']
            elif 'PID' in cjdata:
                pid_arr = cjdata['PID']
            else:
                pid_arr = "unknown"
                
            if isinstance(pid_arr, np.ndarray):
                if pid_arr.dtype.kind in ['i', 'u']:
                    pid = "".join([chr(int(c)) for c in pid_arr.flatten() if c > 0]).strip()
                else:
                    pid = "".join([str(c) for c in pid_arr.flatten()]).strip()
            else:
                pid = str(pid_arr).strip()
                
            if not pid:
                pid = "unknown"
                
            return label, pid
        except Exception as e_sio:
            raise ValueError(f"Failed to read metadata. h5py error: {e_h5py}; scipy error: {e_sio}")

def extract_mat_data(file_path):
    """Extracts image, label, and patient ID from a MAT file supporting Matlab 7.3 (h5py) or older (scipy.io.loadmat)."""
    try:
        with h5py.File(file_path, 'r') as f:
            if 'cjdata' in f:
                cjdata = f['cjdata']
                image = np.array(cjdata['image']).T
                label = int(np.array(cjdata['label']).flatten()[0])
                
                pid_arr = np.array(cjdata['PID']).flatten()
                if pid_arr.dtype.kind in ['U', 'S']:
                    pid = "".join([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in pid_arr]).strip()
                else:
                    pid = "".join(chr(int(c)) for c in pid_arr if c > 0).strip()
            else:
                image = np.array(f['image']).T
                label = int(np.array(f['label']).flatten()[0])
                pid_arr = np.array(f['PID']).flatten()
                if pid_arr.dtype.kind in ['U', 'S']:
                    pid = "".join([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in pid_arr]).strip()
                else:
                    pid = "".join(chr(int(c)) for c in pid_arr if c > 0).strip()
            
            if not pid:
                pid = "unknown"
            return image, label, pid
    except Exception as e_h5py:
        try:
            mat_data = sio.loadmat(file_path)
            if 'cjdata' in mat_data:
                cjdata = mat_data['cjdata'][0, 0]
            else:
                cjdata = mat_data
                
            if hasattr(cjdata, 'dtype') and 'label' in cjdata.dtype.names:
                label = int(cjdata['label'].flatten()[0])
            elif 'label' in cjdata:
                label = int(cjdata['label'].flatten()[0])
            else:
                raise KeyError("Could not find 'label' field.")
                
            if hasattr(cjdata, 'dtype') and 'PID' in cjdata.dtype.names:
                pid_arr = cjdata['PID']
            elif 'PID' in cjdata:
                pid_arr = cjdata['PID']
            else:
                pid_arr = "unknown"
                
            if isinstance(pid_arr, np.ndarray):
                if pid_arr.dtype.kind in ['i', 'u']:
                    pid = "".join([chr(int(c)) for c in pid_arr.flatten() if c > 0]).strip()
                else:
                    pid = "".join([str(c) for c in pid_arr.flatten()]).strip()
            else:
                pid = str(pid_arr).strip()
                
            if not pid:
                pid = "unknown"
                
            if hasattr(cjdata, 'dtype') and 'image' in cjdata.dtype.names:
                image = cjdata['image']
            elif 'image' in cjdata:
                image = cjdata['image']
            else:
                raise KeyError("Could not find 'image' field.")
                
            return image, label, pid
        except Exception as e_sio:
            raise ValueError(f"Failed to read MAT file. h5py error: {e_h5py}; scipy error: {e_sio}")

def process_mat_files(mat_files_or_dir):
    """Parses MAT files, performs patient-grouped splits, and saves PNGs by class."""
    print("[*] process_mat_files: Initializing MAT processor...")
    if isinstance(mat_files_or_dir, str):
        if not os.path.exists(mat_files_or_dir):
            print(f"[-] process_mat_files failed: Directory {mat_files_or_dir} not found.")
            return
        mat_files = [os.path.join(mat_files_or_dir, f) for f in os.listdir(mat_files_or_dir) if f.endswith(".mat")]
    else:
        mat_files = [f for f in mat_files_or_dir if f.endswith(".mat")]
        
    print(f"[+] process_mat_files: Found {len(mat_files)} MAT files. Starting MAT validation (metadata extraction)...")
    
    labels = []
    pids = []
    file_paths = []
    
    for i, file_path in enumerate(mat_files):
        try:
            label, pid = extract_mat_metadata(file_path)
            labels.append(label)
            pids.append(pid)
            file_paths.append(file_path)
            if (i + 1) % 500 == 0 or (i + 1) == len(mat_files):
                print(f"  - Successfully extracted metadata for {i + 1}/{len(mat_files)} MAT files...")
        except Exception as e:
            print(f"  [-] Error reading MAT file {os.path.basename(file_path)}: {e}")
            
    print(f"[+] process_mat_files: Validation complete. Parsed {len(file_paths)}/{len(mat_files)} MAT files.")
    if len(file_paths) == 0:
        raise ValueError("No MAT files could be parsed. All files failed to load.")
        
    labels = np.array(labels)
    pids = np.array(pids)
    file_paths = np.array(file_paths)
    
    print("[*] process_mat_files: Mapping labels...")
    if labels.min() == 1:
        print(f"    - Meningioma (Class 1): {np.sum(labels == 1)}")
        print(f"    - Glioma (Class 2): {np.sum(labels == 2)}")
        print(f"    - Pituitary (Class 3): {np.sum(labels == 3)}")
        labels = labels - 1
    else:
        print(f"    - Meningioma (Class 0): {np.sum(labels == 0)}")
        print(f"    - Glioma (Class 1): {np.sum(labels == 1)}")
        print(f"    - Pituitary (Class 2): {np.sum(labels == 2)}")
    print(f"    - Total patients: {len(np.unique(pids))}")
    
    from sklearn.model_selection import train_test_split
    print("[*] process_mat_files: Performing patient-grouped stratified train/val/test splits (70/10/20)...")
    
    # Extract unique patient IDs and their corresponding diagnosis labels
    unique_pids, unique_indices = np.unique(pids, return_index=True)
    patient_labels = labels[unique_indices]
    
    print(f"    - Total unique patients: {len(unique_pids)}")
    print(f"    - Patient-level label distribution:")
    print(f"      - Meningioma (Class 0): {np.sum(patient_labels == 0)}")
    print(f"      - Glioma (Class 1): {np.sum(patient_labels == 1)}")
    print(f"      - Pituitary (Class 2): {np.sum(patient_labels == 2)}")
    
    # Split patient IDs into train, val, and test splits (70/10/20) while stratifying on patient diagnosis
    train_pids, temp_pids, _, temp_pat_lbls = train_test_split(
        unique_pids, patient_labels, test_size=0.3, random_state=42, stratify=patient_labels
    )
    val_pids, test_pids, _, _ = train_test_split(
        temp_pids, temp_pat_lbls, test_size=0.6667, random_state=42, stratify=temp_pat_lbls
    )
    
    # Map back from patient IDs to slice files and slice labels
    train_mask = np.isin(pids, train_pids)
    val_mask = np.isin(pids, val_pids)
    test_mask = np.isin(pids, test_pids)
    
    train_files = file_paths[train_mask]
    train_lbls = labels[train_mask]
    val_files = file_paths[val_mask]
    val_lbls = labels[val_mask]
    test_files = file_paths[test_mask]
    test_lbls = labels[test_mask]
    
    splits = {
        'train': (train_files, train_lbls),
        'val': (val_files, val_lbls),
        'test': (test_files, test_lbls)
    }
    
    print(f"    - Train split size: {len(train_files)} slices (from {len(train_pids)} patients)")
    print(f"    - Val split size: {len(val_files)} slices (from {len(val_pids)} patients)")
    print(f"    - Test split size: {len(test_files)} slices (from {len(test_pids)} patients)")
    
    class_names = {0: "meningioma", 1: "glioma", 2: "pituitary"}
    
    print("[*] process_mat_files: Creating folder structure under output directory...")
    if os.path.exists(OUTPUT_DIR):
        try:
            shutil.rmtree(OUTPUT_DIR)
            print(f"[+] Cleared old output directory: {OUTPUT_DIR}")
        except Exception as e:
            print(f"[-] Warning: Failed to clean old OUTPUT_DIR: {e}")
        
    for split in ['train', 'val', 'test']:
        for label_idx in class_names.values():
            os.makedirs(os.path.join(OUTPUT_DIR, split, label_idx), exist_ok=True)
            
    print("[*] process_mat_files: Exporting MAT image matrices to PNG...")
    for split_name, (paths, lbls) in splits.items():
        print(f"  - Converting '{split_name}' split ({len(paths)} images)...")
        for idx, (path, lbl) in enumerate(zip(paths, lbls)):
            try:
                img, _, _ = extract_mat_data(path)
                img = img.astype(np.float32)
                
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img_norm = 255.0 * (img - img_min) / (img_max - img_min)
                else:
                    img_norm = img * 0
                img_norm = img_norm.astype(np.uint8)
                
                pil_img = Image.fromarray(img_norm)
                class_folder = class_names[lbl]
                file_id = os.path.basename(path).replace(".mat", ".png")
                save_path = os.path.join(OUTPUT_DIR, split_name, class_folder, file_id)
                pil_img.save(save_path)
                
                if (idx + 1) % 200 == 0 or (idx + 1) == len(paths):
                    print(f"    - [{split_name}] Converted and saved {idx + 1}/{len(paths)} PNGs...")
            except Exception as e:
                print(f"    [-] Error converting image {os.path.basename(path)}: {e}")
                
    final_count = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        final_count += sum(1 for f in files if f.endswith(".png"))
    print(f"[+] process_mat_files: Successfully exported {final_count} PNG files to {OUTPUT_DIR}.")

def find_and_process_input(force_clear=False):
    import os
    import zipfile
    import shutil
    from collections import Counter
    
    print("[*] STEP 1: Initializing find_and_process_input...")
    input_dir = "/kaggle/input"
    if not os.path.exists(input_dir):
        input_dir = "."
    print(f"[*] STEP 2: Input directory set to: {input_dir}")
    
    global OUTPUT_DIR
    OUTPUT_DIR = "dataset_processed"
    print(f"[*] STEP 3: Output directory set to: {OUTPUT_DIR}")
    
    # Check if dataset is already preprocessed and complete
    if not force_clear and os.path.exists(OUTPUT_DIR):
        png_count = 0
        for root, dirs, files in os.walk(OUTPUT_DIR):
            png_count += sum(1 for f in files if f.endswith(".png"))
        if png_count >= 3000:
            print(f"[+] STEP 4: Found {png_count} already preprocessed PNG files in {OUTPUT_DIR}. Skipping preprocessing to save time.")
            return
            
    # Force clearing the cache if requested or if incomplete
    print("[*] STEP 4: Clearing/resetting processed dataset cache...")
    if os.path.exists(OUTPUT_DIR):
        try:
            shutil.rmtree(OUTPUT_DIR)
            print("[+] Successfully cleared existing dataset_processed folder.")
        except Exception as e:
            print(f"[-] Warning: Failed to delete {OUTPUT_DIR}: {e}")
    else:
        print("[*] No existing dataset_processed folder found. Ready for a clean run.")
        
    print("[*] STEP 5: Scanning input directory recursively for Cheng dataset (up to depth 5)...")
    mat_files = []
    zip_files = []
    
    base_depth = os.path.abspath(input_dir).count(os.sep)
    for root, dirs, files in os.walk(input_dir):
        # Skip hidden and output directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'output', 'outputs', 'dataset_processed']]
        
        # Limit depth to 5
        depth = os.path.abspath(root).count(os.sep) - base_depth
        if depth >= 5:
            dirs[:] = []
            
        for f in files:
            path = os.path.join(root, f)
            if f.endswith(".mat"):
                name_part = f[:-4]
                # Check if it is a numeric filename between 1 and 3064
                if name_part.isdigit() and 1 <= int(name_part) <= 3064:
                    mat_files.append(path)
            elif f.endswith(".zip"):
                zip_files.append(path)
                
        # Early exit if we found the complete dataset (3064 slices)
        if len(mat_files) >= 3064:
            print(f"  - Found all {len(mat_files)} Cheng dataset MAT files. Stopping scan early.")
            break
            
    print(f"[*] STEP 6: Scan complete. Found {len(mat_files)} matching Cheng MAT files and {len(zip_files)} total ZIP files.")
    
    if len(mat_files) >= 3000:
        print(f"[+] STEP 7: Unzipped Cheng dataset detected ({len(mat_files)} files). Starting processing...")
        process_mat_files(mat_files)
        return
        
    if len(zip_files) > 0:
        print("[*] STEP 7: Zip files detected. Identifying Cheng dataset main zip...")
        main_zip = None
        for z in zip_files:
            try:
                print(f"  - Checking zip contents: {z}...")
                with zipfile.ZipFile(z, 'r') as zf:
                    names = zf.namelist()
                    if any("brainTumorDataPublic" in n for n in names) or any(n.endswith(".mat") for n in names):
                        main_zip = z
                        print(f"    [+] Found Cheng dataset zip: {z}")
                        break
            except Exception as e:
                print(f"    [-] Error checking zip {z}: {e}")
                pass
                
        if main_zip:
            print(f"[+] STEP 8: Extracting main dataset zip: {main_zip}...")
            mat_dir = extract_dataset(main_zip)
            if mat_dir:
                mat_files_extracted = []
                for root, dirs, files in os.walk(mat_dir):
                    for f in files:
                        if f.endswith(".mat"):
                            name_part = f[:-4]
                            if name_part.isdigit() and 1 <= int(name_part) <= 3064:
                                mat_files_extracted.append(os.path.join(root, f))
                print(f"[+] Extracted {len(mat_files_extracted)} MAT files. Starting processing...")
                process_mat_files(mat_files_extracted)
                return
            else:
                print("[-] Error: Extraction failed to return a MAT directory.")
        else:
            public_zips = [z for z in zip_files if "brainTumorDataPublic" in os.path.basename(z)]
            if len(public_zips) > 0:
                print(f"[+] STEP 8: Found {len(public_zips)} public brain tumor zip segments directly. Extracting...")
                temp_extract = "mat_files_temp"
                if os.path.exists(temp_extract):
                    shutil.rmtree(temp_extract)
                os.makedirs(temp_extract, exist_ok=True)
                for z in public_zips:
                    print(f"  - Extracting nested zip segment: {z}...")
                    with zipfile.ZipFile(z, 'r') as zip_ref:
                        zip_ref.extractall(temp_extract)
                mat_files_extracted = []
                for root, dirs, files in os.walk(temp_extract):
                    for f in files:
                        if f.endswith(".mat"):
                            name_part = f[:-4]
                            if name_part.isdigit() and 1 <= int(name_part) <= 3064:
                                mat_files_extracted.append(os.path.join(root, f))
                print(f"[+] Extracted {len(mat_files_extracted)} MAT files in temp folder. Starting processing...")
                process_mat_files(mat_files_extracted)
                shutil.rmtree(temp_extract)
                return

    raise ValueError("Cheng dataset not found in input directory! Please attach the Cheng Figshare brain tumor dataset to your Kaggle notebook.")


if __name__ == '__main__':
    find_and_process_input(force_clear=True)
