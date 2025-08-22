הנה גרסה מסודרת ונקייה יותר ל-README שלך:

---

# 🚑 Injury Prediction Dashboard

מערכת דשבורד מבוססת **Streamlit** לחיזוי פציעות בתאונות דרכים, בעזרת מודל CatBoost וניתוח מרחבי.

---

## 📂 מבנה התיקייה

```
DASHBOARD_305325417_...
│
├── app.py                # קובץ ההרצה הראשי של הדשבורד
├── catboost_model.pkl    # המודל המאומן
├── processed_data.csv    # נתונים מעובדים לשימוש
├── kepler_map.html       # מפה אינטראקטיבית מסוג Kepler
├── my_folium_map.html    # מפה אינטראקטיבית מסוג Folium
├── requirements.txt      # ספריות נדרשות להרצה
```

---

## ⚙ התקנה והרצה

1. ודאו שאתם נמצאים בתיקיית הפרויקט (אותה תיקייה בה נמצאים `app.py` ו־`requirements.txt`).

2. התקינו את כל הספריות הדרושות (הרצה חד־פעמית):

   ```bash
   pip install -r requirements.txt
   ```

3. הריצו את האפליקציה באמצעות Streamlit:

   ```bash
   python -m streamlit run app.py
   ```

4. לאחר מכן תופיע כתובת מקומית (בדרך כלל: [http://localhost:8501](http://localhost:8501)) – לחצו עליה כדי לראות את הדשבורד.

---
