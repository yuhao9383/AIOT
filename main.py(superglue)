import asyncio
import json
import sys, os
from pathlib import Path

# 匯入 SuperGlue 模型
sys.path.append(os.path.join(os.path.dirname(__file__), 'module'))
from module.superglue import init_model, run_matching
from module.websocket import connect_and_handshake, listen_one
from module.pnp import run_solvepnp_from_json

# === 狀態變數 ===
# requesting_coordinate 控制 send_request_coordinate 是否持續送出 request_json
# awaiting_response    控制避免重複發送 request_json
requesting_coordinate = True
awaiting_response = False
first_capture = True

matching, device = init_model()

# === 重置，初始化 ===
def reset_serial_number():
    execution_path = Path(__file__).resolve().parent.parent / "execution.json"
    with open(execution_path, "w", encoding="utf-8") as f:
        json.dump({"serial_numbers": 1}, f, indent=4, ensure_ascii=False)
    print("🔄 已將 serial_number 重置為 1")

# === 訊息解析 ===
def process_message(message: str):
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        print("訊息格式錯誤，無法解析")
        return None

    notification = data.get("notification")
    if notification:
        print(f"收到通知: {notification}")
        return notification
    else:
        print("訊息中未包含 notification 欄位")
        return None

# === 定時請求 JSON ===
async def send_request_coordinate(websocket):
    global requesting_coordinate, awaiting_response
    while True:
        if requesting_coordinate and not awaiting_response:
            request_msg = json.dumps({"action": "request_json"})
            await websocket.send(request_msg)
            awaiting_response = True    # 標記已發送，等回應後才再次發送
            print(f"送出請求訊息：{request_msg}")
        await asyncio.sleep(3)

# === SuperGlue 處理 ===
def run_superglue(matching, device):
    execution_path = Path(__file__).resolve().parent.parent / "execution.json"
    with open(execution_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    serial_number = str(config['serial_numbers'])

    base_path = Path(__file__).resolve().parent.parent / "data_base" / serial_number
    input_dir = base_path / "b"
    output_dir = base_path / "c"

    img0 = input_dir / "respiberry.jpg"
    img1 = input_dir / "cesium.png"

    run_matching(matching, device, img0, img1,
                 enable_viz=True, top_k='all', output_dir=output_dir)

# === 處理回應 ===
async def handle_message(result: str, websocket):
    global requesting_coordinate, awaiting_response, first_capture

    match result:
        case "has_coordinate":
            print("有座標，等待下一輪")
            awaiting_response = False
            # 暫時關閉自動發送 request_json
            requesting_coordinate = False

            # 觸發 Cesium 重載
            renew = json.dumps({"action": "renew_cesium"})
            await websocket.send(renew)
            print(f"送出 renew_cesium：{renew}")

            if first_capture:
                # 第一次 capture 時暫停 15 秒
                print("初始化中....，暫停 15 秒等待 Cesium 載入")
                await asyncio.sleep(15)
                first_capture = False

            # 15 秒後重新開放自動發送
            requesting_coordinate = True
            print("重新開放 request_json 發送")

            # === 更新 serial_number 並準備下一輪 ===
            execution_path = Path(__file__).resolve().parent.parent / "execution.json"
            with open(execution_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            config["serial_numbers"] += 1
            with open(execution_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"🔁 序號更新為 {config['serial_numbers']}，開始下一輪")

        case "no_coordinate":
            print("沒有座標，請求 Cesium 畫面")
            requesting_coordinate = False
            awaiting_response = False
            msg = json.dumps({"action": "get_cesium_picture"})
            await websocket.send(msg)
            print(f"送出圖片請求：{msg}")
            await asyncio.sleep(1.5)

        case "got_cesium_picture":
            print("收到 got_cesium_picture，開始匹配")
            await websocket.send(json.dumps({"action": "status_update", "step": "superglue"}))
            run_superglue(matching, device)
            msg = json.dumps({"action": "request_coordinate"})
            await websocket.send(msg)
            awaiting_response = True
            print(f"✅ 匹配完成，送出 request_coordinate：{msg}")

        case "got_match_world_coordinates":
            print("開始進行 PnP 配對")
            awaiting_response = False
            requesting_coordinate = False

            # 讀取 serial_number
            execution_path = Path(__file__).resolve().parent.parent / "execution.json"
            with open(execution_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            serial_number = str(config["serial_numbers"])

            # 組 match JSON 路徑
            base = Path(__file__).resolve().parent.parent
            match_path = base / "data_base" / serial_number / "c" / "respiberry_cesium_matches.json"

            try:
                result = run_solvepnp_from_json(str(match_path))
            except Exception as e:
                print(f"解算時發生錯誤：{e}")
                result = None

            if result is None:
                print("❌ PnP 解算失敗，使用上一筆座標")
                prev_sn = config["serial_numbers"] - 1
                prev_info_path = base / "data_base" / str(prev_sn) / "a" / "flight_information.json"
                lat = lon = height = 0.0
                heading = pitch = roll = None
                if prev_info_path.exists():
                    with open(prev_info_path, "r", encoding="utf-8") as f:
                        prev_info = json.load(f)
                        lat = prev_info.get("latitude", 0.0)
                        lon = prev_info.get("longitude", 0.0)
                        height = prev_info.get("height", 0.0)
                        heading = prev_info.get("heading")
                        pitch = prev_info.get("pitch")
                        roll = prev_info.get("roll")

                await websocket.send(json.dumps({"action": "status_update", "step": "calculation_done"}))
                await websocket.send(json.dumps({
                    "action": "calculation_result",
                    "latitude": lat,
                    "longitude": lon,
                    "height": height,
                    "heading": heading,
                    "pitch": pitch,
                    "roll": roll,
                    "status": "failed",
                    "note": "解算失敗"
                }))

                info_path = base / "data_base" / serial_number / "a" / "flight_information.json"
                info_path.parent.mkdir(parents=True, exist_ok=True)
                if info_path.exists():
                    with open(info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                else:
                    info = {}

                info.update({
                    "latitude": lat,
                    "longitude": lon,
                    "height": height,
                    "heading": heading,
                    "pitch": pitch,
                    "roll": roll,
                    "calculated": False,
                    "status": "failed",
                    "note": "解算失敗"
                })
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=4, ensure_ascii=False)
                print(f"📝 已寫入失敗定位結果：{info_path}")

            else:
                lat, lon, height = result
                print(f"相機 WGS84 位置：緯度={lat:.6f}, 經度={lon:.6f}, 高度={height:.2f}m")
                await websocket.send(json.dumps({"action": "status_update", "step": "calculation_done"}))
                await websocket.send(json.dumps({
                    "action": "calculation_result",
                    "latitude": lat,
                    "longitude": lon,
                    "height": height,
                    "status": "ok"
                }))

                info_path = base / "data_base" / serial_number / "a" / "flight_information.json"
                info_path.parent.mkdir(parents=True, exist_ok=True)
                if info_path.exists():
                    with open(info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                else:
                    info = {}

                info.update({
                    "latitude": lat,
                    "longitude": lon,
                    "height": height,
                    "calculated": True,
                    "status": "ok"
                })
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=4, ensure_ascii=False)
                print(f"📝 已寫入定位結果：{info_path}")

            # === 更新 serial_number 並準備下一輪 ===
            config["serial_numbers"] += 1
            with open(execution_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"🔁 序號更新為 {config['serial_numbers']}，開始下一輪")
            requesting_coordinate = True

        case None:
            print("未能解析訊息，略過")
            awaiting_response = False

        case other:
            print(f"收到未知通知: {other}")
            awaiting_response = False

# === 主程式 ===
async def main():
    uri = "ws://localhost:8080"
    websocket = await connect_and_handshake(uri)
    try:
        task = asyncio.create_task(send_request_coordinate(websocket))
        while True:
            msg = await listen_one(websocket)
            print(f"收到訊息：{msg}")
            res = process_message(msg)
            await handle_message(res, websocket)
    except KeyboardInterrupt:
        print("手動停止")
    finally:
        task.cancel()
        await websocket.close()

if __name__ == '__main__':
    reset_serial_number()
    asyncio.run(main())
