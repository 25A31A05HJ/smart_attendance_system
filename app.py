from flask import Flask, render_template, request, jsonify
from database import create_tables, get_connection
from face_engine import recognize_student

import base64
import cv2
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

create_tables()


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# REGISTER STUDENT
# ==========================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        student_id = request.form["student_id"]
        name = request.form["name"]
        department = request.form["department"]

        face_image = request.files.get("face_image")

        if not face_image:
            return render_template(
                "register.html",
                message="Please select a face image."
            )

        # Make sure faces folder exists
        os.makedirs("faces", exist_ok=True)

        # Save face image
        filename = student_id + ".jpg"
        filepath = os.path.join("faces", filename)

        face_image.save(filepath)

        connection = get_connection()

        try:

            connection.execute("""
                INSERT INTO students
                (student_id, name, department, face_file)
                VALUES (?, ?, ?, ?)
            """, (
                student_id,
                name,
                department,
                filepath
            ))

            connection.commit()

            message = "Student registered successfully!"

        except Exception as error:

            message = f"Error: {error}"

        finally:

            connection.close()

        return render_template(
            "register.html",
            message=message
        )

    return render_template("register.html")


# ==========================================
# SCANNER PAGE
# ==========================================

@app.route("/scanner")
def scanner():

    return render_template("scanner.html")


# ==========================================
# RECOGNIZE FACE
# ==========================================

@app.route("/recognize", methods=["POST"])
def recognize():

    try:

        data = request.get_json()

        if not data or "image" not in data:

            return jsonify({
                "message": "❌ No image received."
            })


        image_data = data["image"]

        # Remove Base64 header
        image_data = image_data.split(",")[1]

        # Decode image
        image_bytes = base64.b64decode(image_data)

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return jsonify({
                "message": "❌ Could not read image."
            })


        # Recognize face
        student_id = recognize_student(image)


        if student_id is None:

            return jsonify({
                "message": "❌ Face not recognized."
            })


        # Find student
        connection = get_connection()

        student = connection.execute("""
            SELECT *
            FROM students
            WHERE student_id = ?
        """, (
            student_id,
        )).fetchone()


        if student is None:

            connection.close()

            return jsonify({
                "message": "❌ Student not found."
            })


        # Current date and time
        now = datetime.now()

        current_date = now.strftime(
            "%Y-%m-%d"
        )

        current_time = now.strftime(
            "%H:%M:%S"
        )


        # Check whether attendance already exists
        existing = connection.execute("""
            SELECT *
            FROM attendance
            WHERE student_id = ?
            AND date = ?
        """, (
            student_id,
            current_date
        )).fetchone()


        if existing:

            connection.close()

            return jsonify({
                "message":
                f"⚠️ {student['name']} "
                "is already marked present today."
            })


        # Save attendance
        connection.execute("""
            INSERT INTO attendance
            (student_id, date, time, status)
            VALUES (?, ?, ?, ?)
        """, (
            student_id,
            current_date,
            current_time,
            "Present"
        ))

        connection.commit()

        connection.close()


        return jsonify({
            "message":
            f"✅ Attendance marked for "
            f"{student['name']} "
            f"at {current_time}"
        })


    except Exception as error:

        print("ERROR:", error)

        return jsonify({
            "message":
            "❌ Error processing face."
        })


# ==========================================
# ATTENDANCE PAGE
# ==========================================

@app.route("/attendance")
def attendance():

    connection = get_connection()

    records = connection.execute("""
        SELECT
            attendance.student_id,
            students.name,
            attendance.date,
            attendance.time,
            attendance.status

        FROM attendance

        LEFT JOIN students

        ON attendance.student_id =
           students.student_id

        ORDER BY
            attendance.date DESC,
            attendance.time DESC
    """).fetchall()

    connection.close()

    return render_template(
        "attendance.html",
        records=records
    )


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)