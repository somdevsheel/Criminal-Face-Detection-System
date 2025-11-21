### 1. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt


## 📖 Usage

### Running the Web Application

```bash
python -m src.api.app
```

The application will be available at `http://localhost:5000`

### Using the Web Interface

The dashboard displays real-time statistics:
- **Total Subjects**: Number of registered criminals in the database
- **Total Events**: Total recognition attempts logged
- **Recent Matches**: Number of successful matches

#### Navigation Tabs

1. **Recognize Face** (Default):
   - Upload an image using "Choose file" button
   - Adjust the similarity threshold slider (default: 0.40)
   - Click "Recognize Faces" to identify individuals
   - View detected faces with match results and confidence scores

2. **Live Stream**:
   - Enable real-time face recognition from webcam
   - Continuous monitoring and detection
   - Instant alerts for database matches
   - Check live stream: http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4
   - Check rtsp stream: rtsp://rtspstream:95yafbaWWMQS5WR8eHVrl@zephyr.rtsp.stream/people
   - Detect using web cam

3. **Process Video**:
   - Upload video files for batch face recognition
   - Process multiple frames automatically
   - Generate comprehensive match reports

4. **Add Subject**:
   - Upload a clear face image
   - Enter Subject ID (e.g., CRIM001)
   - Enter full name
   - Add crime description
   - Save to database with face embeddings

5. **View Subjects**:
   - Browse all registered subjects
   - Search by name, ID, or crime type
   - View subject details and images
   - Edit or delete entries
   - Export subject database

6. **View Events**:
   - Access complete recognition history
   - Filter by event type (MATCH, NO_MATCH, etc.)
   - View timestamps and confidence scores
   - Export event logs for reporting