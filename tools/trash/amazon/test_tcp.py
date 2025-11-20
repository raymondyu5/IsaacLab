import socket

HOST = "0.0.0.0"  # Accept from any interface
PORT = 8888

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"🟢 Server listening on {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()

        with conn:
            while True:
                try:
                    data = conn.recv(1024)
                    if not data:
                        print("❌ Connection closed by client.")
                        break  # Client disconnected
                    message = data.decode()
                    print(f"📥 Received: {message}")

                    # # Optional: Send reply
                    # conn.sendall(b"ack")

                except ConnectionResetError:
                    print("❌ Connection reset by peer.")
                    break
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
                    break
