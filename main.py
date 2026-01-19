import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Configuration
WIDTH, HEIGHT = 1280, 720
FPS_TARGET = 30

class HandTracker:
    """
    Handles hand detection, landmark extraction, and gesture recognition.
    """
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, # Dual hand support
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.img_shape = (720, 1280, 3)

    def find_hands(self, img):
        """
        Processes the image to find hands and returns list of (landmarks, label).
        """
        self.img_shape = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        hands_data = []
        if self.results.multi_hand_landmarks and self.results.multi_handedness:
            for hand_lms, handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                label = handedness.classification[0].label # 'Left' or 'Right'
                hands_data.append((hand_lms, label))
        return hands_data

    def draw_skeleton(self, img, landmarks):
        """
        Draws a sci-fi tech skeleton overlay.
        """
        # Techy colors
        color_bone = (200, 255, 200) # Pale Green/Cyan
        color_joint = (0, 255, 255)  # Cyan
        
        # Connections
        self.mp_draw.draw_landmarks(
            img, landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=color_bone, thickness=2, circle_radius=1),
            self.mp_draw.DrawingSpec(color=color_joint, thickness=2, circle_radius=3)
        )

    def get_palm_center(self, landmarks, hand_state):
        """
        Calculates the center of the palm using Wrist, Index MCP, and Pinky MCP.
        Applies EMA smoothing using hand_state.
        """
        h, w, c = self.img_shape
        
        # Landmark indices: 0=Wrist, 5=Index MCP, 17=Pinky MCP
        wrist = landmarks.landmark[0]
        index_mcp = landmarks.landmark[5]
        pinky_mcp = landmarks.landmark[17]
        
        cx = int((wrist.x + index_mcp.x + pinky_mcp.x) / 3 * w)
        cy = int((wrist.y + index_mcp.y + pinky_mcp.y) / 3 * h)
        
        # EMA Smoothing
        alpha = 0.5
        if hand_state.prev_center is None:
            hand_state.prev_center = (cx, cy)
        else:
            prev_x, prev_y = hand_state.prev_center
            cx = int(alpha * cx + (1 - alpha) * prev_x)
            cy = int(alpha * cy + (1 - alpha) * prev_y)
            hand_state.prev_center = (cx, cy)
            
        return (cx, cy)

    def is_palm_open(self, landmarks):
        """
        Checks if the palm is open based on finger tips vs pip joints distance.
        Checks if at least 4 fingers are extended.
        """
        # Finger tip indices: 8, 12, 16, 20 (Index, Middle, Ring, Pinky)
        # Finger pip indices: 6, 10, 14, 18
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        open_fingers = 0
        wrist = landmarks.landmark[0]
        
        for i in range(4):
            tip = landmarks.landmark[tips[i]]
            pip = landmarks.landmark[pips[i]]
            
            dist_tip = (tip.x - wrist.x)**2 + (tip.y - wrist.y)**2
            dist_pip = (pip.x - wrist.x)**2 + (pip.y - wrist.y)**2
            
            if dist_tip > dist_pip:
                open_fingers += 1

        # Check Thumb
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        if (thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2 > (thumb_ip.x - wrist.x)**2 + (thumb_ip.y - wrist.y)**2:
             open_fingers += 1

        return open_fingers >= 4

    def detect_thrust(self, landmarks, hand_state):
        """
        Detects a forward thrust gesture based on Palm Scale.
        Requires 3 consecutive frames of thrust detection.
        """
        wrist = landmarks.landmark[0]
        index_mcp = landmarks.landmark[5]
        
        current_depth_proxy = math.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y)
        
        is_thrust_frame = False
        if hand_state.prev_z_depth is not None:
            delta = current_depth_proxy - hand_state.prev_z_depth
            if delta > 0.05: 
                is_thrust_frame = True
        
        hand_state.prev_z_depth = current_depth_proxy
        
        if is_thrust_frame:
            hand_state.thrust_buffer += 1
        else:
            hand_state.thrust_buffer = max(0, hand_state.thrust_buffer - 1)
            
        return hand_state.thrust_buffer >= 3

    def get_steadiness(self, center, hand_state):
        """
        Calculates movement speed of the palm center.
        Returns True if steady (low movement), False otherwise.
        """
        if hand_state.prev_center_raw is None:
            hand_state.prev_center_raw = center
            return True
        
        prev_x, prev_y = hand_state.prev_center_raw
        curr_x, curr_y = center
        
        dist = math.hypot(curr_x - prev_x, curr_y - prev_y)
        hand_state.prev_center_raw = center
        
        # Threshold for steadiness (pixels per frame)
        return dist < 8.0
    
    def detect_pinch(self, landmarks):
        """Returns True if Index Tip and Thumb Tip are close."""
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        
        # Calculate distance
        dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        return dist < 0.06 # Increased from 0.04 for easier pinch
    
    def detect_pointing(self, landmarks):
        """Returns True if only Index finger is open."""
        # Check fingers open:
        wrist = landmarks.landmark[0]
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        open_fingers = []
        for i in range(4):
            tip = landmarks.landmark[tips[i]]
            pip = landmarks.landmark[pips[i]]
            dist_tip = (tip.x - wrist.x)**2 + (tip.y - wrist.y)**2
            dist_pip = (pip.x - wrist.x)**2 + (pip.y - wrist.y)**2
            open_fingers.append(dist_tip > dist_pip)
            
        # Index open (True), others closed (False)
        # We allow thumb to be whatever
        return open_fingers[0] and not open_fingers[1] and not open_fingers[2] and not open_fingers[3]

class ParticleSystem:
    """
    Manages the anti-gravity energy particles.
    """
    def __init__(self):
        self.particles = [] # List of [x, y, vx, vy, life, color, size, offset_angle]
        self.max_particles = 80

    def emit_burst(self, x, y, count=10):
        """Spawns an explosion of particles."""
        for _ in range(count):
            if len(self.particles) >= self.max_particles: break
            
            px = x + np.random.randint(-5, 5)
            py = y + np.random.randint(-5, 5)
            # Explosion velocity: Radial
            angle = np.random.uniform(0, 6.28)
            speed = np.random.uniform(2, 6)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            
            life = 1.0
            color = (0, 255, 255) # Cyan spark
            size = np.random.randint(2, 6)
            offset_angle = 0
            self.particles.append([px, py, vx, vy, life, color, size, offset_angle])

    def emit(self, x, y, charge_level=0.5):
        """Spawns new particles."""
        if len(self.particles) >= self.max_particles:
            return

        # Slight randomness in position
        px = x + np.random.randint(-15, 15)
        py = y + np.random.randint(-15, 15)
        
        # Velocity: random horizontal, negative vertical (upwards)
        vx = np.random.uniform(-1, 1)
        vy = np.random.uniform(-4, -1) # Upwards
        
        life = 1.0 # 100% life
        
        # Color based on charge: Blue/Cyan -> White
        # BGR
        blue = 255
        green = 200 + int(55 * charge_level)
        red = int(255 * charge_level)
        color = (blue, green, red)
        
        size = np.random.randint(2, 5)
        offset_angle = np.random.uniform(0, 6.28)
        
        self.particles.append([px, py, vx, vy, life, color, size, offset_angle])

    def update(self):
        """Updates physics for all particles."""
        for p in self.particles:
            # Add spiral motion
            p[7] += 0.1 # Angle
            spiral_x = math.sin(p[7]) * 0.5
            
            p[0] += p[2] + spiral_x # x += vx + spiral
            p[1] += p[3] # y += vy
            p[4] -= 0.02 # life decay
            
            p[2] += np.random.uniform(-0.1, 0.1)

        # Remove dead particles
        self.particles = [p for p in self.particles if p[4] > 0]

    def draw(self, img):
        """Draws particles on the image."""
        for p in self.particles:
            x, y = int(p[0]), int(p[1])
            life = p[4]
            size = p[6]
            base_color = p[5]
            
            # Fade out
            color = (
                int(base_color[0] * life), 
                int(base_color[1] * life), 
                int(base_color[2] * life)
            )
            
            cv2.circle(img, (x, y), size, color, -1)

class RepulsorEffect:
    """
    Manages the state and rendering of the Repulsor beam.
    """
    def __init__(self):
        self.state = "IDLE" # IDLE, CHARGING, FULL_CHARGE, FIRING, COOLDOWN
        self.charge_level = 0.0
        self.cooldown = 0
        self.max_charge_radius = 60
    
    def update(self, palm_center, is_hand_present, is_palm_open, is_thrust, is_steady):
        """Updates the state machine."""
        if self.cooldown > 0:
            self.cooldown -= 1
            if self.cooldown == 0:
                self.state = "IDLE"
            else:
                self.state = "COOLDOWN"
            return 
        
        if not is_hand_present:
            self.charge_level = max(0, self.charge_level - 0.02) # Slow decay
            self.state = "IDLE"
            return
            
        if is_palm_open:
            if is_thrust and self.charge_level >= 0.8:
                self.state = "FIRING"
                self.cooldown = 20 # Frames of cooldown
                self.charge_level = 0.0 # Reset
            
            elif self.charge_level < 1.0:
                 if is_steady:
                     self.state = "CHARGING"
                     self.charge_level += 0.015 # Charge rate
                 else:
                     # If moving too fast, pause charging or slight decay
                     self.state = "CHARGING" # Still show state, but maybe don't increase?
                     # Let's just allow it but maybe particles look chaotic
                     pass
            else:
                 self.charge_level = 1.0
                 self.state = "FULL_CHARGE"
        else:
            self.charge_level = max(0, self.charge_level - 0.02) # Slow decay
            self.state = "IDLE"

    def draw(self, img, palm_center, wrist_pos=None):
        """Renders the repulsor visual effects."""
        x, y = palm_center
        
    def draw(self, img, palm_center, wrist_pos=None):
        """Renders the repulsor visual effects."""
        x, y = palm_center
        base_radius = int(20 + self.charge_level * 40)
        
        # 1. Glow Effect (Layered)
        if self.charge_level > 0:
            # Calculate glow radius based on charge
            base_radius = int(20 + self.charge_level * 40)
            
            # Layer 1: Outer (Blue, Faint)
            overlay = img.copy()
            cv2.circle(overlay, (x, y), int(base_radius * 1.5), (255, 100, 0), -1) 
            cv2.addWeighted(overlay, 0.2 * self.charge_level, img, 1.0, 0, img)
            
            # Layer 2: Middle (Cyan, Bright)
            overlay = img.copy()
            cv2.circle(overlay, (x, y), int(base_radius * 1.2), (255, 200, 0), -1)
            cv2.addWeighted(overlay, 0.4 * self.charge_level, img, 1.0, 0, img)
            
        # 2. State-Specific Visuals
        if self.state in ["CHARGING", "FULL_CHARGE"]:
            # Inner White Core with jitter if high charge
            jitter = np.random.randint(-2, 2) if self.state == "FULL_CHARGE" else 0
            cv2.circle(img, (x+jitter, y+jitter), int(15 + self.charge_level*10), (255, 255, 255), -1)
            
            # Rotating Rings
            radius = int(base_radius)
            angle = time.time() * 5
            # Draw ring segments could be cool, but simple circles for now
            cv2.circle(img, (x, y), radius, (255, 255, 200), 2)
            cv2.circle(img, (x, y), radius - 5, (255, 200, 0), 1)
            
        elif self.state == "FIRING":
            # Radial Beam Logic
            # Direction: Wrist -> Palm
            if wrist_pos:
                wx, wy = wrist_pos
                dx, dy = x - wx, y - wy
                mag = math.hypot(dx, dy)
                if mag > 0:
                    dx, dy = dx/mag, dy/mag
                else:
                    dx, dy = 0, -1 # Default up
                
                # Beam End Point (off screen)
                end_x = int(x + dx * 1000)
                end_y = int(y + dy * 1000)
                
                # Draw Beam
                # Core
                cv2.line(img, (x, y), (end_x, end_y), (255, 255, 255), 40)
                # Outer Glow
                overlay = img.copy()
                cv2.line(overlay, (x, y), (end_x, end_y), (255, 200, 0), 80) # Cyan
                cv2.line(overlay, (x, y), (end_x, end_y), (255, 100, 0), 120) # Blueish
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
                
                # Flash at palm
                cv2.circle(img, (x, y), 60, (255, 255, 255), -1)
            else:
                 # Fallback if no wrist (shouldn't happen with tracker)
                 cv2.circle(img, (x, y), 80, (255, 255, 255), -1)

        elif self.state == "IDLE":
             # Small idle indicator
             cv2.circle(img, (x, y), 5, (200, 200, 200), 1)

class HandState:
    """Holds state for a single hand."""
    def __init__(self):
        self.prev_center = None        # Smoothed center
        self.prev_center_raw = None    # Raw center for steadiness
        self.prev_z_depth = None
        self.thrust_buffer = 0         # Number of consecutive thrust frames
        self.repulsor = RepulsorEffect()
        self.is_present = False

class FaceTracker:
    """
    Handles Face Mesh detection for the Iron Man Helmet HUD.
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks=True gives us iris landmarks (468-477)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_color = (255, 255, 255)
        self.thickness = 2

    def find_face(self, img):
        """Returns the first detected face landmarks."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

    def draw_skeleton(self, img, landmarks):
        """Draws the wireframe mask (tessellation) on the face."""
        if not landmarks: return
        
        # Custom "Tech" Style: Cyan lines, no dots
        draw_spec_lines = self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=0)
        
        self.mp_draw.draw_landmarks(
            image=img,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=draw_spec_lines
        )

    def draw_hud(self, img, landmarks):
        """Draws the J.A.R.V.I.S. Interface on the face."""
        if not landmarks: return
        h, w, _ = img.shape
        points = [(int(l.x * w), int(l.y * h)) for l in landmarks.landmark]
        
        # Colors
        cyan = (255, 255, 0)
        
        # 1. Eye Reticles (Iris centers: 468 Left, 473 Right)
        if len(points) > 473:
            left_iris = points[468]
            right_iris = points[473]
            
            # Left Eye
            cv2.circle(img, left_iris, 25, cyan, 1)
            cv2.line(img, (left_iris[0]-20, left_iris[1]), (left_iris[0]-35, left_iris[1]), cyan, 1)
            cv2.line(img, (left_iris[0]+20, left_iris[1]), (left_iris[0]+35, left_iris[1]), cyan, 1)
            
            # Right Eye
            cv2.circle(img, right_iris, 25, cyan, 1)
            cv2.line(img, (right_iris[0]-20, right_iris[1]), (right_iris[0]-35, right_iris[1]), cyan, 1)
            cv2.line(img, (right_iris[0]+20, right_iris[1]), (right_iris[0]+35, right_iris[1]), cyan, 1)
            
            # Text Specs
            cv2.putText(img, "ID: SS001", (right_iris[0]+40, right_iris[1]-10), cv2.FONT_HERSHEY_PLAIN, 0.8, cyan, 1)

        # 2. Forehead Scan
        forehead = points[10]
        cv2.putText(img, "L.A.N.D. SYSTEM ONLINE", (forehead[0]-100, forehead[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 3. Chin / Jawline elements
        chin = points[152]
        cv2.putText(img, "ARMOR INTEGRITY: 100%", (chin[0]-80, chin[1]+40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 255), 1)
        
        # Tech Lines (Cheekbones)
        left_cheek = points[234]
        right_cheek = points[454]
        cv2.line(img, left_cheek, (left_cheek[0]-20, left_cheek[1]+20), cyan, 1)
        cv2.line(img, right_cheek, (right_cheek[0]+20, right_cheek[1]+20), cyan, 1)

class FloatingObject:
    """A virtual object created by the user (3D Hologram)."""
    def __init__(self, shape_type, points, center):
        self.shape_type = shape_type # 'circle', 'rect'
        self.center = np.array([center[0], center[1]], dtype=float) 
        self.velocity = np.array([0.0, 0.0])
        self.color = (0, 255, 255) # Yellow/Gold default
        self.is_grabbed = False
        self.radius = 40
        self.angle_x = 0
        self.angle_y = 0
        self.angle_z = 0
        
        # Visuals (Juice)
        self.trail = [] # List of (x, y)
        self.stress_color = self.color
        
        if shape_type == 'circle':
             self.radius = int(cv2.minEnclosingCircle(np.array(points))[1])
        elif shape_type == 'rect':
             x,y,w,h = cv2.boundingRect(np.array(points))
             self.radius = max(w, h) // 2 # Approximation for collision
             
        # Clamp Size (Fix for "Too Big")
        self.radius = max(20, min(self.radius, 60))

    def update(self, particles=None):
        """Apply Physics and Rotation."""
        if not self.is_grabbed:
            # Apply Friction/Drag
            self.velocity *= 0.95 
            self.center += self.velocity
            
            speed = np.linalg.norm(self.velocity)
            
            # Boundary Bounce with Particle Effect
            hit_wall = False
            if self.center[0] < 0 or self.center[0] > WIDTH: 
                self.velocity[0] *= -1
                hit_wall = True
            if self.center[1] < 0 or self.center[1] > HEIGHT: 
                self.velocity[1] *= -1
                hit_wall = True
                
            if hit_wall and speed > 10 and particles:
                particles.emit_burst(self.center[0], self.center[1], count=10)
        
        # Trail Logic
        if np.linalg.norm(self.velocity) > 2 or self.is_grabbed:
            self.trail.append(tuple(self.center))
            if len(self.trail) > 10: self.trail.pop(0)
        elif len(self.trail) > 0:
            self.trail.pop(0) # Fade out when stopped

        # Auto Rotate (Speed depends on velocity)
        rot_speed = 0.02 + (np.linalg.norm(self.velocity) * 0.005)
        self.angle_x += rot_speed
        self.angle_y += rot_speed + 0.01

    def draw_3d(self, img):
        # Update Color based on Velocity (Stress)
        speed = np.linalg.norm(self.velocity)
        if speed > 20: self.stress_color = (0, 0, 255) # Red
        elif speed > 10: self.stress_color = (0, 165, 255) # Orange
        else: self.stress_color = self.color # Gold/Cyan
        
        # Draw Trail
        if len(self.trail) > 1:
            pts = np.array(self.trail, np.int32)
            # Fading line? CV2 polylines is solid.
            # Let's draw varying thickness/alpha manually or just simple line for now specific color
            cv2.polylines(img, [pts], False, self.stress_color, 1, cv2.LINE_AA)

        if self.shape_type == 'rect':
            self.draw_cube(img)
        elif self.shape_type == 'circle':
            self.draw_sphere(img)

    def draw_cube(self, img):
        # 3D Cube Vertices centered at 0,0,0
        r = self.radius * 0.7 # Scale down slightly
        vertices = np.array([[-r, -r, -r], [r, -r, -r], [r, r, -r], [-r, r, -r],
                             [-r, -r, r], [r, -r, r], [r, r, r], [-r, r, r]])
        
        # Rotation Matrix (Y and X axes)
        ax, ay = self.angle_x, self.angle_y
        
        rot_x = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
        rot_y = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
        
        # Project and Draw
        projected_points = []
        for v in vertices:
            rotated = np.dot(rot_y, np.dot(rot_x, v))
            # Simple Orthographic + offset
            projected_points.append([int(rotated[0] + self.center[0]), int(rotated[1] + self.center[1])])
        
        # Edges
        edges = [(0,1), (1,2), (2,3), (3,0),
                 (4,5), (5,6), (6,7), (7,4),
                 (0,4), (1,5), (2,6), (3,7)]
                 
        for s, e in edges:
            pt1 = tuple(projected_points[s])
            pt2 = tuple(projected_points[e])
            cv2.line(img, pt1, pt2, self.stress_color, 2)
            
    def draw_sphere(self, img):
        cx, cy = int(self.center[0]), int(self.center[1])
        r = int(self.radius)
        c = self.stress_color
        
        # Outer rim
        cv2.circle(img, (cx, cy), r, c, 2)
        
        # Lat/Long lines (Ellipses)
        # Rotate based on angle
        angle = int(math.degrees(self.angle_y)) % 360
        
        # Simulate rotation by changing ellipse width
        width_ratio = math.cos(self.angle_y)
        cv2.ellipse(img, (cx, cy), (r, int(r * abs(width_ratio))), 0, 0, 360, c, 1)
        cv2.ellipse(img, (cx, cy), (int(r * abs(width_ratio)), r), 0, 0, 360, c, 1)
        
        # Core
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)

    def contains(self, px, py):
        dx = px - self.center[0]
        dy = py - self.center[1]
        # Magnetic Grab: Radius + 20px buffer
        return math.hypot(dx, dy) < self.radius + 30

class Fabricator:
    """Manages Air Drawing and Object Creation."""
    def __init__(self):
        self.drawing_path = []
        self.objects = []
        self.grabbed_object = None
        self.last_draw_time = 0
        self.prev_pinch_pos = None # For calculating velocity
        self.prev_hand_pos = (0, 0) # For hand velocity (pushing)
    
    def update(self, img, index_tip, is_pointing, is_pinching, particles=None):
        x, y = index_tip
        hand_velocity = np.array([x - self.prev_hand_pos[0], y - self.prev_hand_pos[1]])
        self.prev_hand_pos = (x, y)
        
        # 1. Update Objects (Physics)
        hovering_any = False
        
        for obj in self.objects:
            obj.update(particles)
            
            # Smart Interaction: Check Proximity
            dist_to_hand = math.hypot(x - obj.center[0], y - obj.center[1])
            hit_radius = obj.radius + 30
            
            if dist_to_hand < hit_radius:
                hovering_any = True
                
                # PHYSICS PUSH (If not grabbing and hand moving fast enough)
                if not is_pinching and np.linalg.norm(hand_velocity) > 5:
                    # Apply impulse
                    push_dir = hand_velocity * 1.5
                    obj.velocity += push_dir
                    
        # 2. Drawing Mode (with Inhibition)
        # Only draw if NOT hovering over an object (Prioritize Interaction)
        if is_pointing and not is_pinching and not hovering_any:
            # Add point if simplified
            if len(self.drawing_path) == 0 or math.hypot(x - self.drawing_path[-1][0], y - self.drawing_path[-1][1]) > 5:
                self.drawing_path.append((x, y))
                self.last_draw_time = time.time()
                
        # Shape Recog (Same as before but sets type)
        elif len(self.drawing_path) > 10 and (time.time() - self.last_draw_time) > 0.5:
             points = np.array(self.drawing_path)
             M = cv2.moments(points)
             if M["m00"] != 0:
                 cX = int(M["m10"] / M["m00"])
                 cY = int(M["m01"] / M["m00"])
                 
                 obj_type = 'circle' # default
                 x_rect,y_rect,w,h = cv2.boundingRect(points)
                 # aspect_ratio = float(w)/h
                 peri = cv2.arcLength(points, True)
                 approx = cv2.approxPolyDP(points, 0.04 * peri, True)
                 
                 # Lenient Rect Detection (3-6 vertices -> Cube)
                 # This captures Sloppy Triangles, Squares, Pentagons as Cubes
                 if 3 <= len(approx) <= 6:
                      obj_type = 'rect' # Cube
                 
                 self.objects.append(FloatingObject(obj_type, points, [cX, cY]))
             self.drawing_path = []

        # 3. Manipulation with Physics Throwing
        if is_pinching:
            if self.grabbed_object:
                # Move
                self.grabbed_object.center = np.array([float(x), float(y)])
                self.grabbed_object.velocity = np.array([0.0, 0.0]) # Zero velocity while holding
                
                # Calculate Throw Velocity (Momentum History)
                # Just use hand velocity directly? - Yes simpler
                
                # Store simple velocity for release
                # We calculate release velocity at the moment of release usually
                pass
            else:
                # Try grab
                for obj in self.objects:
                    if obj.contains(x, y):
                        self.grabbed_object = obj
                        self.grabbed_object.is_grabbed = True
                        self.grabbed_object.velocity = np.array([0.0, 0.0])
                        self.prev_pinch_pos = (x, y)
                        break
        else:
            if self.grabbed_object:
                # Release Throw
                if self.prev_pinch_pos:
                     # Use the hand velocity we calculated at top of frame for consistency?
                     # Or delta since last pinched pos?
                     # Let's use current instantaneous hand velocity for throw
                     self.grabbed_object.velocity = hand_velocity * 0.8 # Throw factor
                     
                self.grabbed_object.is_grabbed = False
                self.grabbed_object = None
            self.prev_pinch_pos = None
            
        # Draw Objects
        for obj in self.objects:
            obj.draw_3d(img)
            
        # Draw Path
        if len(self.drawing_path) > 1:
            # Smooth Polyline
            pts = np.array(self.drawing_path, np.int32)
            cv2.polylines(img, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)

class HUDManager:
    """Manages Virtual Buttons & Helmet Overlay."""
    def __init__(self):
        self.buttons = [] # [x, y, w, h, text, callback]
    
    def add_button(self, text, x, y, callback):
        # Increased size for hex
        self.buttons.append({'text': text, 'center': (x, y), 'radius': 40, 'callback': callback})
        
    def draw_hex(self, img, center, radius, color, thickness=2, fill=False):
        cx, cy = center
        pts = []
        for i in range(6):
            angle_deg = 60 * i + 30
            angle_rad = math.radians(angle_deg)
            px = int(cx + radius * math.cos(angle_rad))
            py = int(cy + radius * math.sin(angle_rad))
            pts.append([px, py])
        pts = np.array(pts, np.int32)
        if fill:
            cv2.fillPoly(img, [pts], color)
        else:
            cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
            
    def update(self, img, pointer_pos, is_clicking):
        # 1. Global Visor Overlay
        h, w = img.shape[:2]
        cyan = (255, 255, 0)
        
        # Corner Brackets
        l = 40
        t = 2
        # TL
        cv2.line(img, (20, 20), (20+l, 20), cyan, t)
        cv2.line(img, (20, 20), (20, 20+l), cyan, t)
        # TR
        cv2.line(img, (w-20, 20), (w-20-l, 20), cyan, t)
        cv2.line(img, (w-20, 20), (w-20, 20+l), cyan, t)
        # BL
        cv2.line(img, (20, h-20), (20+l, h-20), cyan, t)
        cv2.line(img, (20, h-20), (20, h-20-l), cyan, t)
        # BR
        cv2.line(img, (w-20, h-20), (w-20-l, h-20), cyan, t)
        cv2.line(img, (w-20, h-20), (w-20, h-20-l), cyan, t)
        
        # Center Crosshair (Faint)
        cv2.line(img, (w//2 - 10, h//2), (w//2 + 10, h//2), (200, 255, 200), 1)
        cv2.line(img, (w//2, h//2 - 10), (w//2, h//2 + 10), (200, 255, 200), 1)

        # 2. Buttons
        px, py = pointer_pos
        for btn in self.buttons:
            bx, by = btn['center']
            r = btn['radius']
            
            # Hit Test (Hexagon approx as Circle)
            dist = math.hypot(px - bx, py - by)
            is_hover = dist < r
            
            color = (0, 165, 255) # Orange default
            if is_hover:
                color = (0, 255, 0) # Green Hover
                if is_clicking:
                    btn['callback']()
                    self.draw_hex(img, (bx, by), r, (255, 255, 255), fill=True) # Flash

            self.draw_hex(img, (bx, by), r, color, 2)
            # Center Text
            font_size = 0.6
            text = btn['text']
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
            cv2.putText(img, text, (bx - tw//2, by + th//2), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)

def main():
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam. Please check your camera connection.")
            return

        cap.set(3, WIDTH)
        cap.set(4, HEIGHT)
        
        tracker = HandTracker()
        face_tracker = FaceTracker()
        particles = ParticleSystem()
        
        # v5 Systems
        fabricator = Fabricator()
        hud_manager = HUDManager()
        
        # Mode State
        mode = "COMBAT" # or "FABRICATION"
        
        def toggle_mode():
            nonlocal mode
            if mode == "COMBAT": mode = "FABRICATION"
            else: mode = "COMBAT"
            
        # UI: Move to Right Side (WIDTH - 100)
        # Using a fixed X for now assuming width ~1280 or standard
        # Better to use WIDTH relative if variable, but here we hardcode relative to known size
        hud_manager.add_button("MODE", WIDTH - 100, 100, toggle_mode)
        
        # State tracking for Left and Right hands
        hand_states = {
            "Left": HandState(),
            "Right": HandState()
        }
        
        prev_time = 0

        print("Iron Man Repulsor System Initiated...")
        print("Controls: 'q' or 'ESC' to quit, 'd' for debug, 'r' to reset.")
        
        debug_mode = False

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame. Exiting...")
                break

            # Mirror effect
            img = cv2.flip(img, 1)
            
            try:
                # 1. Hand Detection
                detected_hands = tracker.find_hands(img)
                
                # 2. Face Detection & HUD
                face_landmarks = face_tracker.find_face(img)
                if face_landmarks:
                    face_tracker.draw_skeleton(img, face_landmarks)
                    face_tracker.draw_hud(img, face_landmarks)
                
                # Reset presence for this frame
                for state in hand_states.values():
                    state.is_present = False
                
                # Process each detected hand
                primary_pointer = None # For HUD interaction
                is_clicking_hud = False
                
                for landmarks, label in detected_hands:
                    
                    if label in hand_states:
                        state = hand_states[label]
                        state.is_present = True
                        
                        # Draw Skeleton
                        tracker.draw_skeleton(img, landmarks)
                        
                        # Logic
                        palm_center = tracker.get_palm_center(landmarks, state)
                        wrist = landmarks.landmark[0]
                        wrist_pos = (int(wrist.x * WIDTH), int(wrist.y * HEIGHT))
                        index_tip = landmarks.landmark[8]
                        ix, iy = int(index_tip.x * WIDTH), int(index_tip.y * HEIGHT)
                        
                        is_palm_open = tracker.is_palm_open(landmarks)
                        is_thrust = tracker.detect_thrust(landmarks, state)
                        is_steady = tracker.get_steadiness(palm_center, state)
                        is_pinch = tracker.detect_pinch(landmarks)
                        is_point = tracker.detect_pointing(landmarks)
                        
                        if label == "Right": # Right hand controls UI/Drawing primarily
                            primary_pointer = (ix, iy)
                            is_clicking_hud = is_pinch
                        
                        # MODE SWITCHING LOGIC
                        if mode == "COMBAT":
                            # Update Repulsor
                            state.repulsor.update(palm_center, True, is_palm_open, is_thrust, is_steady)
                            
                            # Emit Particles (every 2 frames for perf)
                            if state.repulsor.state in ["CHARGING", "FULL_CHARGE"]:
                                if int(time.time() * 30) % 2 == 0: # Half rate
                                    particles.emit(palm_center[0], palm_center[1], state.repulsor.charge_level)
                                
                            # Draw Repulsor
                            state.repulsor.draw(img, palm_center, wrist_pos)
                            
                            # HUD / Debug Info specific to hand
                            hud_x = palm_center[0] + 50 if label == "Right" else palm_center[0] - 150
                            hud_y = palm_center[1]
                            
                            status_color = (255, 255, 255) # White
                            if state.repulsor.state == "CHARGING": status_color = (255, 200, 0) # Cyan
                            if state.repulsor.state == "FULL_CHARGE": status_color = (0, 0, 255) # Red/Orange
                            if state.repulsor.state == "FIRING": status_color = (255, 255, 255)
                            
                            cv2.putText(img, f"STATUS: {state.repulsor.state}", (hud_x, hud_y), cv2.FONT_HERSHEY_PLAIN, 1, status_color, 1)
                            if state.repulsor.state == "CHARGING":
                                 cv2.rectangle(img, (hud_x, hud_y+10), (hud_x + int(100*state.repulsor.charge_level), hud_y+15), (255, 200, 0), -1)

                        elif mode == "FABRICATION":
                            # Draw text
                            cv2.putText(img, "FABRICATION MODE", (WIDTH//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(img, "Point to Draw. Pinch to Grab.", (WIDTH//2 - 150, 80), cv2.FONT_HERSHEY_PLAIN, 1, (200, 255, 200), 1)
                            
                            # Update Fabricator (Only Right Hand for now)
                            if label == "Right":
                                fabricator.update(img, (ix, iy), is_point, is_pinch, particles)


                # Update states for hands NOT present
                for label, state in hand_states.items():
                    if not state.is_present:
                        state.repulsor.update((0,0), False, False, False, False)
                        # Clear prev state
                        state.prev_center = None 
                        state.prev_z_depth = None
                        state.thrust_buffer = 0

                # GENERAL UPDATES
                if mode == "COMBAT":
                     # Update & Draw Particles (Global)
                    particles.update()
                    particles.draw(img)
                elif mode == "FABRICATION":
                    # If Fabricator hasn't updated this frame (e.g. no Right Hand), we must still Draw/Update Physics
                    # We can check if `fabricator.last_draw_time` or just simple flag?
                    # Better: Fabricator.draw_all(img) method that is called regardless.
                    
                    # Since Fabricator.update() handles physics + drawing, we need to call it without input if no hand.
                    pass
                
                # Global Drawing for Fabricator overlay
                # We need to draw objects on top regardless of loop
                if mode == "FABRICATION":
                     for obj in fabricator.objects:
                         if not any(obj is fabricator.objects[i] for i in range(len(fabricator.objects)) if False): # Just checking
                             pass 
                         # We need to make sure we don't double draw if update was called.
                         # Actually, update() draws. 
                         # Let's just iterate and draw if they weren't drawn? 
                         # Simpler: Move draw out of update().
                         pass
                    
                     # Hack: If no right hand, force update physics/draw
                     right_present = hand_states["Right"].is_present
                     if not right_present:
                         # Update physics with no input
                         for obj in fabricator.objects:
                            obj.update(particles) # Add particles
                            obj.draw_3d(img)
                         # Draw path
                         if len(fabricator.drawing_path) > 1:
                            pts = np.array(fabricator.drawing_path, np.int32)
                            cv2.polylines(img, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)

                # HUD Manager (Global)
                if primary_pointer:
                    hud_manager.update(img, primary_pointer, is_clicking_hud)
                
                # FPS Calculation
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
                prev_time = curr_time
                
                # Global UI
                cv2.putText(img, "Somyajeet Singh", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if debug_mode:
                    cv2.putText(img, f"FPS: {int(fps)}", (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    
            except Exception as e:
                print(f"Error in processing loop: {e}")
                import traceback
                traceback.print_exc()

            cv2.imshow("Iron Man Repulsor", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
            elif key == ord('r'):
                for state in hand_states.values():
                    state.repulsor.charge_level = 0.0
                    


        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...") # Keep terminal open to see error

if __name__ == "__main__":
    main()
