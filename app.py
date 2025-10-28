# ================================
# College Marketplace – CSV Edition
# ================================
# - CSV-only: users.csv, chats.csv, faculty_books.csv, listings.csv
# - WhatsApp-style chat UI
# - Local advisor bot (no ML/APIs)
# - Faculty approvals + Faculty Picks tab
# - 🛒 Sell/Buy Marketplace for Seniors
# ================================

# -----------------------------
# Imports & page config
# -----------------------------
import os
import re
import uuid
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="College Marketplace – CSV Edition", layout="wide")

# WhatsApp style chat UI CSS
st.markdown("""
<style>
.chat-container {
    max-height: 350px;
    overflow-y: auto;
    padding: 10px;
    background-color: #e5ddd5;
    border-radius: 12px;
}
.message {
    padding: 8px 12px;
    border-radius: 18px;
    margin: 6px;
    max-width: 75%;
    display: inline-block;
}
.me {
    background-color: #DCF8C6;
    float: right;
    clear: both;
}
.bot {
    background-color: white;
    float: left;
    clear: both;
}
.timestamp {
    font-size: 10px;
    color: #777;
    text-align: right;
}
.card {
    border-radius: 12px;
    padding: 10px;
    background: #fafafa;
    border: 1px solid #eee;
}
.badge {
    display:inline-block;
    padding:2px 8px;
    border-radius:10px;
    background:#eef;
    font-size:12px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CSV constants & helpers
# -----------------------------
USERS_CSV = "users.csv"          # user_id,name,role,department,semester
CHATS_CSV = "chats.csv"          # chat_id,sender_id,receiver_id,message,timestamp
FACULTY_CSV = "faculty_books.csv" # book_id,approved_by,timestamp
LISTINGS_CSV = "listings.csv"    # listing_id,seller_id,book_name,author_name,price,condition,notes,timestamp,status

BOT_USER_ID = "bot"
BOT_NAME = "Advisor Bot"

def ensure_csv_files():
    """Create seed CSVs if they don't exist."""
    if not os.path.exists(USERS_CSV):
        seed = pd.DataFrame([
            {"user_id": "u1", "name": "Aarav (Junior)", "role": "junior", "department": "CSE", "semester": 2},
            {"user_id": "u2", "name": "Diya (Senior)",  "role": "senior", "department": "CSE", "semester": 7},
            {"user_id": "u3", "name": "Rahul (Senior)", "role": "senior", "department": "ECE", "semester": 8},
            {"user_id": "u4", "name": "Meera (Faculty)","role": "faculty","department": "CSE","semester": None},
        ])
        seed.to_csv(USERS_CSV, index=False)

    if not os.path.exists(CHATS_CSV):
        pd.DataFrame(columns=["chat_id","sender_id","receiver_id","message","timestamp"]).to_csv(CHATS_CSV, index=False)

    if not os.path.exists(FACULTY_CSV):
        pd.DataFrame(columns=["book_id","approved_by","timestamp"]).to_csv(FACULTY_CSV, index=False)

    if not os.path.exists(LISTINGS_CSV):
        pd.DataFrame(columns=[
            "listing_id","seller_id","book_name","author_name","price","condition","notes","timestamp","status"
        ]).to_csv(LISTINGS_CSV, index=False)

@st.cache_data(show_spinner=False)
def load_users() -> pd.DataFrame:
    ensure_csv_files()
    df = pd.read_csv(USERS_CSV)
    if "semester" in df.columns:
        df["semester"] = pd.to_numeric(df["semester"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_chats() -> pd.DataFrame:
    ensure_csv_files()
    try:
        return pd.read_csv(CHATS_CSV)
    except Exception:
        return pd.DataFrame(columns=["chat_id","sender_id","receiver_id","message","timestamp"])

def append_chat(sender_id: str, receiver_id: str, message: str):
    """Append a chat message to chats.csv and clear cache."""
    row = {
        "chat_id": str(uuid.uuid4()),
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "message": message.strip(),
        "timestamp": datetime.utcnow().isoformat()
    }
    df = load_chats().copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CHATS_CSV, index=False)
    load_chats.clear()  # invalidate cache

def load_faculty_books():
    ensure_csv_files()
    return pd.read_csv(FACULTY_CSV)

def approve_book(book_id):
    entry = {
        "book_id": int(book_id),
        "approved_by": st.session_state["current_user"],
        "timestamp": datetime.utcnow().isoformat()
    }
    df = load_faculty_books()
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(FACULTY_CSV, index=False)

@st.cache_data(show_spinner=False)
def load_listings():
    ensure_csv_files()
    return pd.read_csv(LISTINGS_CSV)

def save_listings(df):
    df.to_csv(LISTINGS_CSV, index=False)
    load_listings.clear()

# -----------------------------
# Books dataset loader (uploaded CSV)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_books(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame(columns=["Book Name","Author Name","Rating","Price"])  # skeleton
    try:
        df = pd.read_csv(file)
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, sep=';', encoding='latin-1')

    # Normalize required columns
    need = ["Book Name","Author Name","Rating","Price"]
    rename_map = {}
    for want in need:
        if want not in df.columns:
            for c in df.columns:
                if c.strip().lower().replace(" ","") == want.lower().replace(" ",""):
                    rename_map[c] = want
                    break
    if rename_map:
        df = df.rename(columns=rename_map)
    for col in need:
        if col not in df.columns:
            df[col] = np.nan

    # Clean numeric fields
    df["Rating"] = df["Rating"].astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)").astype(float)
    df["Price"]  = (
        df["Price"].astype(str).str.replace(",","",regex=False)
          .str.extract(r"([0-9]+(?:\.[0-9]+)?)").astype(float)
    )

    df = df.dropna(how="all").drop_duplicates(subset=["Book Name","Author Name"], keep="first")
    df["book_id"] = range(1, len(df)+1)
    return df

# -----------------------------
# Lightweight recommendation engine (no sklearn)
# -----------------------------
def get_similar(df: pd.DataFrame, seed_title: str, top_k: int = 10) -> pd.DataFrame:
    """Simplified similar recommendations: same author first, else top-rated."""
    if df.empty:
        return df

    seed_title_lower = seed_title.strip().lower()
    seed_author = df.loc[df["Book Name"].str.lower() == seed_title_lower, "Author Name"].head(1).values

    if len(seed_author):
        # Recommend by same author excluding the current book
        same_author = df[
            (df["Author Name"].astype(str).str.lower() == str(seed_author[0]).lower()) &
            (df["Book Name"].astype(str).str.lower() != seed_title_lower)
        ]
        if not same_author.empty:
            return same_author.head(top_k)

    # Fallback: top rated books (excluding the selected one)
    return (
        df[df["Book Name"].astype(str).str.lower() != seed_title_lower]
        .sort_values("Rating", ascending=False)
        .head(top_k)
    )

# -----------------------------
# Rule-based local chatbot (no APIs)
# -----------------------------
def advisor_bot_reply(prompt: str, users_df: pd.DataFrame, books_df: pd.DataFrame) -> str:
    """
    Behaviors:
    - If query mentions department + semester -> suggest top 3 relevant books (rating desc, price asc)
    - If asks for senior/topper/mentor/help -> list up to 3 seniors (same dept if mentioned)
    - If asks 'cheap/cheapest/value/budget' -> top value picks by rating/price
    - Else -> gentle fallback
    """
    text = (prompt or "").lower()

    # dept detection
    depts = sorted(users_df["department"].dropna().astype(str).str.upper().unique().tolist())
    dept = None
    for d in depts:
        if d.lower() in text:
            dept = d
            break

    # semester detection
    sem = None
    m = re.search(r"sem(?:ester)?\s*(\d+)", text)
    if m:
        try:
            sem = int(m.group(1))
        except Exception:
            sem = None

    # senior help
    if any(kw in text for kw in ["senior", "topper", "mentor", "help", "guide"]):
        pool = users_df[users_df["role"].str.lower()=="senior"].copy()
        if dept:
            pool = pool[pool["department"].astype(str).str.upper()==dept]
        if pool.empty:
            return "I couldn't find any seniors in your department yet. Try adding some to users.csv."
        pool = pool.head(3)
        recs = [f"• {r['name']} — Dept: {str(r['department'])}, Sem: {str(int(r['semester'])) if not pd.isna(r['semester']) else '-'}"
                for _, r in pool.iterrows()]
        return "Here are seniors who can help:\n" + "\n".join(recs)

    # dept + semester → book picks
    if dept and sem is not None and not books_df.empty:
        subset = books_df.copy()
        if dept in ["CSE", "CS", "COMPUTER", "IT"]:
            subset = subset[subset["Book Name"].str.contains("computer|data|algorithm|program|python|java|ml|ai", case=False, na=False)]
        elif dept in ["ECE", "EEE", "ELECTRONICS"]:
            subset = subset[subset["Book Name"].str.contains("circuit|signal|analog|digital|communication|vlsi|micro", case=False, na=False)]
        elif dept in ["ME", "MECH", "MECHANICAL"]:
            subset = subset[subset["Book Name"].str.contains("mechanics|thermo|fluid|machine|design", case=False, na=False)]

        subset = subset.sort_values(["Rating","Price"], ascending=[False, True]).head(3)
        if subset.empty:
            return "I don't see department-specific titles in your CSV. Upload a richer dataset or broaden keywords."

        lines = [
            f"• {r['Book Name']} — {r['Author Name']} "
            f"(⭐ {r['Rating']:.1f}, ₹{int(r['Price']) if not pd.isna(r['Price']) else 'NA'})"
            for _, r in subset.iterrows()
        ]
        return f"Recommended for {dept} Semester {sem}:\n" + "\n".join(lines)

    # best value picks
    if any(kw in text for kw in ["cheap", "cheapest", "value", "budget", "low price"]):
        tmp = books_df.copy()
        tmp = tmp[(tmp["Rating"].notna()) & (tmp["Price"].notna())]
        if tmp.empty:
            return "I need rating and price columns to compute value picks."
        tmp["value_score"] = tmp["Rating"] / (tmp["Price"] + 1e-9)
        top = tmp.sort_values("value_score", ascending=False).head(3)
        lines = [
            f"• {r['Book Name']} — {r['Author Name']} "
            f"(⭐ {r['Rating']:.1f}, ₹{int(r['Price'])})"
            for _, r in top.iterrows()
        ]
        return "Top value picks:\n" + "\n".join(lines)

    return (
        "I'm a campus advisor bot using only your local CSV data. "
        "Try: 'Which book for CSE semester 2?', 'Which senior can help with IoT?', "
        "or 'show best value picks'."
    )

# -----------------------------
# UI helpers
# -----------------------------
def amazon_link(book_name: str) -> str:
    q = re.sub(r"\s+","+", (book_name or "").strip())
    return f"https://www.amazon.in/s?k={q}"

def flipkart_link(book_name: str) -> str:
    q = re.sub(r"\s+","+", (book_name or "").strip())
    return f"https://www.flipkart.com/search?q={q}"

def display_book_card(row: pd.Series, key_prefix: str = "", users_df: pd.DataFrame = None):
    col1, col2, col3, col4 = st.columns([4, 3, 2, 3])
    with col1:
        st.markdown(f"**{row.get('Book Name','N/A')}**")
        st.caption(row.get("Author Name", "Unknown Author"))
        r = row.get("Rating", np.nan)
        p = row.get("Price", np.nan)
        rtxt = f"{r:.1f}" if pd.notna(r) else "NA"
        ptxt = f"₹{int(p)}" if pd.notna(p) else "NA"
        st.write(f"⭐ {rtxt} · {ptxt}")
    with col2:
        st.link_button("View on Amazon", amazon_link(row.get("Book Name","")), use_container_width=True)
        st.link_button("View on Flipkart", flipkart_link(row.get("Book Name","")), use_container_width=True)
    with col3:
        if st.button("Bookmark", key=f"{key_prefix}bm_{row['book_id']}"):
            st.session_state.setdefault("bookmarks", set()).add(int(row["book_id"]))
            st.toast("Added to bookmarks", icon="✅")
        if st.button("Add Note", key=f"{key_prefix}note_{row['book_id']}"):
            st.session_state.setdefault("notes_open", set()).add(int(row["book_id"]))
    with col4:
        role = users_df.loc[users_df["user_id"]==st.session_state["current_user"], "role"].values[0] if users_df is not None else ""
        if role == "faculty":
            if st.button("✅ Approve", key=f"{key_prefix}approve_{row['book_id']}"):
                approve_book(row["book_id"])
                st.toast("Approved for students ✅")
        else:
            cur = float(row["Rating"]) if pd.notna(row["Rating"]) else 0.0
            new_rating = st.slider("Your Rating", 0.0, 5.0, value=cur, step=0.5, key=f"{key_prefix}rate_{row['book_id']}")
            if st.button("Save Rating", key=f"{key_prefix}save_{row['book_id']}"):
                st.session_state.setdefault("user_ratings", {})[int(row["book_id"])] = new_rating
                st.toast("Saved 💾")

    if int(row["book_id"]) in st.session_state.get("notes_open", set()):
        note = st.text_area("Note", key=f"note_text_{row['book_id']}", placeholder="Why this book matters for my course…")
        if st.button("Save Note", key=f"note_save_{row['book_id']}"):
            st.session_state.setdefault("notes", {})[int(row["book_id"])] = note
            st.toast("Saved note", icon="📝")

# ---- Sell/Buy UI helpers ----
def display_listing_card(listing: pd.Series, users_df: pd.DataFrame):
    seller_name = users_df.set_index("user_id").loc[listing["seller_id"], "name"] if listing["seller_id"] in users_df["user_id"].values else listing["seller_id"]
    cols = st.columns([5,2,2,3])
    with cols[0]:
        st.markdown(f"**{listing['book_name']}**")
        st.caption(f"{listing['author_name']} • Seller: {seller_name}")
        st.write(f"₹{int(listing['price']) if pd.notna(listing['price']) else 'NA'}  •  {listing['condition']}")
        if str(listing.get("notes","")).strip():
            st.write(f"📝 {listing['notes']}")
        st.caption(f"ID: {listing['listing_id']} • {listing['timestamp'][:16].replace('T',' ')}")
    with cols[1]:
        if st.button("View on Amazon", key=f"l_amz_{listing['listing_id']}"):
            st.session_state["search_q"] = listing["book_name"]
            st.toast("Open Amazon from Explore for search 🔎")
    with cols[2]:
        if st.button("Message Seller", key=f"l_msg_{listing['listing_id']}"):
            st.session_state["chat_with"] = listing["seller_id"]
            st.toast("Opening chat… 💬")
            st.experimental_rerun()
    with cols[3]:
        is_seller = (st.session_state.get("current_user") == listing["seller_id"])
        if is_seller:
            if listing["status"] != "sold" and st.button("Mark as Sold", key=f"l_sold_{listing['listing_id']}"):
                df = load_listings()
                df.loc[df["listing_id"]==listing["listing_id"], "status"] = "sold"
                save_listings(df)
                st.toast("Marked as sold ✅")
                st.experimental_rerun()
            if st.button("Delete", key=f"l_del_{listing['listing_id']}"):
                df = load_listings()
                df = df[df["listing_id"] != listing["listing_id"]]
                save_listings(df)
                st.toast("Listing deleted 🗑️")
                st.experimental_rerun()
        else:
            st.markdown(f"<span class='badge'>{listing['status'].upper()}</span>", unsafe_allow_html=True)

# -----------------------------
# Sidebar: data & filters & user
# -----------------------------
st.sidebar.header("📥 Load Books CSV")
st.sidebar.caption("Columns needed: Book Name, Author Name, Rating, Price")
books_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
books_df = load_books(books_file)

st.sidebar.header("🔧 Filters")
search = st.sidebar.text_input("Search title/author", "")
if not books_df.empty:
    rmin = float(books_df["Rating"].min(skipna=True)) if books_df["Rating"].notna().any() else 0.0
    rmax = float(books_df["Rating"].max(skipna=True)) if books_df["Rating"].notna().any() else 5.0
    pmax = float(books_df["Price"].max(skipna=True)) if books_df["Price"].notna().any() else 5000.0
else:
    rmin, rmax, pmax = 0.0, 5.0, 1000.0
r1, r2 = st.sidebar.slider("Rating range", 0.0, 5.0, (round(rmin,1), round(min(5.0, rmax),1)), 0.1)
p1, p2 = st.sidebar.slider("Price range (₹)", 0.0, float(max(1000.0, round(pmax,0))), (0.0, float(min(5000.0, round(pmax,0)))), 10.0)
authors = sorted([a for a in books_df["Author Name"].dropna().unique().tolist()]) if not books_df.empty else []
sel_authors = st.sidebar.multiselect("Authors", authors)
sort_by = st.sidebar.selectbox("Sort by", ["Relevance","Rating (desc)","Price (asc)","Price (desc)"])

# user selector (no auth)
users_df = load_users()
user_names = {r["user_id"]: r["name"] for _, r in users_df.iterrows()}
user_ids = list(user_names.keys())

st.sidebar.header("🙋 Current User")
if "current_user" not in st.session_state:
    st.session_state["current_user"] = user_ids[0] if user_ids else ""
cur_user = st.sidebar.selectbox("Choose user", options=user_ids, format_func=lambda i: user_names.get(i, i))
st.session_state["current_user"] = cur_user

# -----------------------------
# Title & KPIs
# -----------------------------
st.title("🎓 College Marketplace – Book Finder, Chat & Advisor (CSV Edition)")
if books_df.empty:
    st.info("Upload your books CSV from the sidebar to unlock recommendations and analytics.")

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Books", f"{len(books_df):,}")
with k2: st.metric("Avg Rating", f"{books_df['Rating'].dropna().mean():.2f}" if not books_df.empty else "0.00")
with k3: st.metric("Avg Price", f"₹{books_df['Price'].dropna().mean():,.0f}" if not books_df.empty else "₹0")
with k4: st.metric("Users", f"{len(users_df):,}")

# -----------------------------
# Filtered dataframe
# -----------------------------
mask = pd.Series(True, index=books_df.index) if not books_df.empty else pd.Series([], dtype=bool)
if not books_df.empty and search.strip():
    s = search.lower().strip()
    mask &= books_df["Book Name"].astype(str).str.lower().str.contains(s) | \
            books_df["Author Name"].astype(str).str.lower().str.contains(s)
if not books_df.empty:
    mask &= books_df["Rating"].fillna(0).between(r1, r2)
    mask &= books_df["Price"].fillna(0).between(p1, p2)
    if sel_authors:
        mask &= books_df["Author Name"].astype(str).isin(sel_authors)

filtered = books_df[mask].copy() if not books_df.empty else books_df.copy()
if sort_by == "Rating (desc)":
    filtered = filtered.sort_values("Rating", ascending=False)
elif sort_by == "Price (asc)":
    filtered = filtered.sort_values("Price", ascending=True)
elif sort_by == "Price (desc)":
    filtered = filtered.sort_values("Price", ascending=False)

# -----------------------------
# Tabs (added Faculty Picks + Sell/Buy)
# -----------------------------
explore_tab, rec_tab, analytics_tab, chat_tab, saved_tab, faculty_tab, market_tab = st.tabs(
    ["🔍 Explore","🤝 Recommendations","📈 Analytics","💬 Chat","⭐ Saved","📘 Faculty Picks","🛒 Sell/Buy"]
)

with explore_tab:
    st.subheader("Results")
    if filtered.empty:
        st.warning("No books match your filters.")
    else:
        for _, row in filtered.iterrows():
            st.divider()
            display_book_card(row, key_prefix="explore_", users_df=users_df)

with rec_tab:
    st.subheader("Pick a seed book for similar suggestions")
    if books_df.empty:
        st.info("Upload books to see recommendations.")
    else:
        seed = st.selectbox("Seed book", options=books_df["Book Name"].tolist())
        k = st.slider("How many?", 5, 20, 10)
        # Lightweight recommender call (no vectorizer/X)
        recs = get_similar(books_df, seed, top_k=k)
        if recs.empty:
            st.info("Not enough data for recommendations yet.")
        else:
            for _, row in recs.iterrows():
                st.divider()
                display_book_card(row, key_prefix="rec_", users_df=users_df)

        st.subheader("💡 Best Value Picks (high rating, lower price)")
        tmp = (filtered if not filtered.empty else books_df).copy()
        tmp = tmp[(tmp["Rating"].notna()) & (tmp["Price"].notna())]
        if tmp.empty:
            st.info("Need rating and price to compute value picks.")
        else:
            tmp["value_score"] = tmp["Rating"] / (tmp["Price"] + 1e-9)
            for _, row in tmp.sort_values("value_score", ascending=False).head(10).iterrows():
                st.divider()
                display_book_card(row, key_prefix="value_", users_df=users_df)

with analytics_tab:
    st.subheader("Top Authors by Count")
    if books_df.empty:
        st.info("Upload books to see analytics.")
    else:
        top_auth = books_df.groupby("Author Name").size().reset_index(name="Count").sort_values("Count", ascending=False).head(15)
        st.bar_chart(top_auth.set_index("Author Name"))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Rating Distribution")
            fig, ax = plt.subplots()
            ax.hist(books_df["Rating"].dropna(), bins=20)
            ax.set_xlabel("Rating")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        with c2:
            st.subheader("Price Distribution (₹)")
            fig, ax = plt.subplots()
            ax.hist(books_df["Price"].dropna(), bins=20)
            ax.set_xlabel("Price (₹)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

with chat_tab:
    st.subheader("Direct Messages & Advisor Bot")

    # Layout: left contacts, center chat
    left, center = st.columns([2, 5])

    with left:
        st.markdown("**Contacts**")
        contacts = users_df[users_df["user_id"] != st.session_state["current_user"]].copy()

        # Add bot as pseudo-contact
        bot_row = pd.DataFrame([[BOT_USER_ID, BOT_NAME, "bot", "", None]],
                               columns=["user_id","name","role","department","semester"])
        contacts = pd.concat([bot_row, contacts], ignore_index=True)

        role_filter = st.selectbox("Filter by role", options=["all","junior","senior","faculty","bot"], index=0)
        if role_filter != "all":
            contacts = contacts[contacts["role"].astype(str).str.lower()==role_filter]

        labels = {r["user_id"]: f"{r['name']} ({r['role']})" for _, r in contacts.iterrows()}
        if "chat_with" not in st.session_state:
            st.session_state["chat_with"] = BOT_USER_ID
        chat_with = st.selectbox("Talk to", options=list(labels.keys()), format_func=lambda i: labels.get(i, i))
        st.session_state["chat_with"] = chat_with

        st.divider()
        st.markdown("**Quick Prompts**")
        if st.button("Which book for CSE semester 2?"):
            st.session_state.setdefault("pending_msg", "Which book for CSE semester 2?")
        if st.button("Which senior can help with IoT?"):
            st.session_state.setdefault("pending_msg", "Which senior can help with IoT?")
        if st.button("Show best value picks"):
            st.session_state.setdefault("pending_msg", "Show best value picks")

    with center:
        st.markdown(f"**Chat with:** {labels.get(st.session_state['chat_with'], st.session_state['chat_with'])}")
        chats_df = load_chats()

        pair = chats_df[
            ((chats_df["sender_id"]==st.session_state["current_user"]) & (chats_df["receiver_id"]==st.session_state["chat_with"])) |
            ((chats_df["sender_id"]==st.session_state["chat_with"]) & (chats_df["receiver_id"]==st.session_state["current_user"]))
        ].sort_values("timestamp")

        # render messages
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for _, m in pair.iterrows():
            who_class = "me" if m["sender_id"]==st.session_state["current_user"] else "bot"
            ts = m["timestamp"].split("T")[1][:5] if isinstance(m["timestamp"], str) and "T" in m["timestamp"] else ""
            st.markdown(
                f"<div class='message {who_class}'>{m['message']}<div class='timestamp'>{ts}</div></div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # input
        default_prefill = st.session_state.pop("pending_msg", "") if "pending_msg" in st.session_state else ""
        msg = st.text_input("Type a message", value=default_prefill, key="chat_input")
        if st.button("Send"):
            if msg.strip():
                append_chat(st.session_state["current_user"], st.session_state["chat_with"], msg)
                if st.session_state["chat_with"] == BOT_USER_ID:
                    reply = advisor_bot_reply(msg, users_df, books_df)
                    append_chat(BOT_USER_ID, st.session_state["current_user"], reply)
                st.rerun()  # safe rerender

with saved_tab:
    st.subheader("Your Bookmarks & Notes")
    bookmarks = st.session_state.get("bookmarks", set())
    notes = st.session_state.get("notes", {})
    user_ratings = st.session_state.get("user_ratings", {})

    if not bookmarks:
        st.info("No bookmarks yet — add a few from Explore or Recommendations.")
    else:
        saved_df = books_df[books_df["book_id"].isin(list(bookmarks))].copy()
        for _, row in saved_df.iterrows():
            st.divider()
            display_book_card(row, key_prefix="saved_", users_df=users_df)
            bid = int(row["book_id"])
            if bid in notes:
                st.caption(f"📝 Note: {notes[bid]}")
            if bid in user_ratings:
                st.caption(f"⭐ Your rating: {user_ratings[bid]}")

        st.download_button(
            "Export bookmarks (CSV)",
            data=saved_df[["Book Name","Author Name","Rating","Price"]].to_csv(index=False).encode("utf-8"),
            file_name="bookmarks.csv",
            mime="text/csv",
        )

with faculty_tab:
    st.subheader("📘 Approved by Faculty")
    approved_df = load_faculty_books()
    if approved_df.empty:
        st.info("No books approved yet.")
    else:
        join_df = approved_df.merge(books_df, on="book_id", how="inner")
        for _, row in join_df.iterrows():
            st.divider()
            display_book_card(row, key_prefix="fac_", users_df=users_df)

# -----------------------------
# 🛒 Sell/Buy Marketplace
# -----------------------------
with market_tab:
    st.subheader("🛒 Sell / Buy Books (Student-to-Student)")

    listings_df = load_listings()

    # Seller form (senior-only)
    role = users_df.loc[users_df["user_id"]==st.session_state["current_user"], "role"].values[0]
    with st.expander("➕ Create a new listing (Seniors only)"):
        if role != "senior":
            st.warning("Only seniors can create listings. Switch to a senior user in the sidebar.")
        else:
            lcol1, lcol2 = st.columns(2)
            with lcol1:
                book_name_in = st.text_input("Book Name")
                author_in = st.text_input("Author Name")
                price_in = st.number_input("Price (₹)", min_value=0, max_value=100000, value=300, step=10)
            with lcol2:
                condition_in = st.selectbox("Condition", ["Like New","Good","Used","Heavily Used"])
                notes_in = st.text_area("Notes (optional)", placeholder="Edition, highlights, meet-up location…")

            if st.button("Post Listing"):
                if str(book_name_in).strip() == "" or str(author_in).strip() == "":
                    st.error("Book name and author are required.")
                else:
                    new_row = {
                        "listing_id": str(uuid.uuid4()),
                        "seller_id": st.session_state["current_user"],
                        "book_name": book_name_in.strip(),
                        "author_name": author_in.strip(),
                        "price": float(price_in),
                        "condition": condition_in,
                        "notes": notes_in.strip(),
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "available"
                    }
                    listings_df = pd.concat([listings_df, pd.DataFrame([new_row])], ignore_index=True)
                    save_listings(listings_df)
                    st.success("Listing posted ✅")

    st.divider()
    st.markdown("### Browse Listings")

    # Filters for marketplace
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        q = st.text_input("Search by title/author", key="m_search")
    with mcol2:
        status_filter = st.selectbox("Status", ["all","available","sold"], index=0)
    with mcol3:
        sort_l = st.selectbox("Sort by", ["Newest","Price (low→high)","Price (high→low)"])

    m_mask = pd.Series(True, index=listings_df.index)
    if q.strip():
        s = q.lower().strip()
        m_mask &= listings_df["book_name"].astype(str).str.lower().str.contains(s) | \
                  listings_df["author_name"].astype(str).str.lower().str.contains(s)
    if status_filter != "all":
        m_mask &= listings_df["status"].astype(str).str.lower() == status_filter

    feed = listings_df[m_mask].copy()
    # sorting
    if sort_l == "Newest":
        feed = feed.sort_values("timestamp", ascending=False)
    elif sort_l == "Price (low→high)":
        feed = feed.sort_values("price", ascending=True)
    else:
        feed = feed.sort_values("price", ascending=False)

    if feed.empty:
        st.info("No listings yet. Seniors can post from the form above.")
    else:
        for _, lst in feed.iterrows():
            st.divider()
            display_listing_card(lst, users_df)

# Footer
st.write("\n")
st.caption("CSV-only demo • Roles: Junior/Senior/Faculty • Local chat & advisor bot • Faculty approvals • Sell/Buy marketplace.")
