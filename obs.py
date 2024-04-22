import simpleobsws

# OBS WebSocket configuration
obs_host = "192.168.1.45"
obs_port = 4455
obs_password = "Dxe7Vf68AhjmOa3h"

ws = simpleobsws.WebSocketClient(url=f"ws://{obs_host}:{obs_port}", password=obs_password)

async def connect_to_obs():
    await ws.connect()
    await ws.wait_until_identified()

async def obs_getSceneItemId(sourceName):
    data = {"sceneName": "Camera Feed", "sourceName": sourceName}
    req = simpleobsws.Request("GetSceneItemId", data)
    ret = await ws.call(req)
    if not ret.ok():
        print('Failed to fetch scene item ID!')
        return False
    return ret.responseData['sceneItemId']

async def switch_obs_scene(cam):
    await ws.call(simpleobsws.Request("SetCurrentProgramScene", {"sceneName": cam['obs']}))
