# 🚀 Deploy Your Poker Range Classifier to Live Web

## 🌟 **Quick Deploy to Render.com (Recommended)**

### **Step 1: Push to GitHub**
```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial commit: Poker Range Classifier"

# Create GitHub repository and push
git remote add origin https://github.com/YOUR_USERNAME/poker-range-classifier.git
git branch -M main
git push -u origin main
```

### **Step 2: Deploy on Render.com**
1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `poker-range-classifier`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn wsgi:app`
   - **Plan**: Free

### **Step 3: Environment Variables**
Add these in Render dashboard:
- `FLASK_ENV`: `production`
- `PORT`: `10000` (Render will set this automatically)

## 🎯 **Alternative: Deploy to Heroku**

### **Step 1: Install Heroku CLI**
```bash
# macOS
brew install heroku/brew/heroku

# Or download from: https://devcenter.heroku.com/articles/heroku-cli
```

### **Step 2: Deploy**
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-poker-app-name

# Deploy
git push heroku main

# Open app
heroku open
```

## 🌐 **Alternative: Deploy to Railway.app**

1. Go to [railway.app](https://railway.app)
2. Connect GitHub repository
3. Auto-deploy with default settings

## 📱 **Custom Domain Setup**

### **Render.com:**
1. Go to your service dashboard
2. Click "Settings" → "Custom Domains"
3. Add your domain (e.g., `poker.yourname.com`)
4. Update DNS records as instructed

### **Heroku:**
```bash
heroku domains:add poker.yourname.com
# Update DNS records as instructed
```

## 🔧 **Troubleshooting**

### **Common Issues:**
- **Model not loading**: Ensure model file is in repository
- **Port issues**: Use `os.environ.get('PORT', 3000)`
- **Dependencies**: Check `requirements.txt` is complete

### **Debug Commands:**
```bash
# Check logs
heroku logs --tail  # Heroku
# Or check Render dashboard logs

# Test locally
python3 app.py
curl http://localhost:3000
```

## 🌟 **Portfolio Benefits**

Your live deployment will show:
- ✅ **Production Deployment Skills**
- ✅ **Full-Stack Development**
- ✅ **ML System Deployment**
- ✅ **Professional Project Management**

## 🎉 **Your Live URL**

After deployment, you'll get:
- **Render**: `https://your-app-name.onrender.com`
- **Heroku**: `https://your-app-name.herokuapp.com`
- **Railway**: `https://your-app-name.railway.app`

**Share this URL in your portfolio and resume!** 🚀
