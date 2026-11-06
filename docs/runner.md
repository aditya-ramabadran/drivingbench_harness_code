# Runner

The **Drive via runner loop** tab (<http://127.0.0.1:8766/?tab=runner>) drives the car from the
laptop gateway, with no chat app. It is a client of the same producer API as the chat tools, so the
comma's controller, limits and stop apply unchanged, and it drives the simulator the same way
([sim.md](sim.md)).

- **A chat model alone** drives with exactly the three public tools (`observe`, `set_motion`,
  `stop_now`, same schemas and wording as the MCP) and the chat apps' prompt.
- **A driver alone**: a decision model on OpenRouter (`typesafe/jev-1.14`,
  `openai/gpt-6-luna-decisions`) picks one of eight steering choices and one of four speeds from
  the latest frame about twice a second and sends a 5 s motion; or a [trained
  policy](#trained-policies) sends a timed chunk about three times a second. Each command is short,
  so the car brakes soon after the loop stalls.
- **Both**: the chat model supervises. It has the same tools plus `set_brief`, which rewrites the
  driver's instructions (for a steerable policy, chooses its next command); `observe` also reports
  the driver's brief, state and latest decisions. Its own `set_motion` pauses the driver until that
  command ends; a stop (its own or the UI's) pauses it until a new brief or **Resume driver**. The
  operator's brake, disengagement or leaving Drive holds it.

## Keys

Keys are read only from the macOS Keychain, never from the environment or a file:

```sh
security add-generic-password -s drivingbench-openrouter -a drivingbench -w   # prompts for the key
security add-generic-password -s drivingbench-gemini -a drivingbench -w       # Google, optional
```

A run asks only for the key it uses: OpenRouter for OpenRouter chat models and decision drivers,
Gemini for `gemini/...` chat models, none for a trained policy alone. `drivingbench doctor` says
which items exist without reading them.

## A run

1. Bring the comma online and engage as for any drive ([install.md](install.md)).
2. Choose the **driver** (none, a decision model, or a trained policy), the **chat model** and its
   **inference tier** (normal, fast or ultrafast, at the provider's rates), and check the
   **objective**. A decision driver takes an optional starting brief, built from the objective
   otherwise. A trained driver takes a chat model only when it is steerable; its objective is
   prefilled from the registry.
3. **Start attempt 1** brings the comma online if needed, checks a trained policy's endpoint once,
   labels a session (e.g. `gpt-6-astra (ultrafast) + jev-1.14`) and starts both layers. Nothing
   opens a session if any of that fails.
4. **End attempt**, with a result (unknown, finished course, operator intervention, collision,
   aborted) and a note, stops the car and sends the chat model the reflection prompt. Between
   attempts only the operator moves the car: model motion is refused with `between_attempts`;
   stopping is always allowed.
5. Once the car is reset, **Next attempt** sends the continuation prompt and restarts the driver
   from the starting brief. Chat models keep their context across attempts.
6. The message box reaches the chat model (e.g. "Continue"). **Stop run** stops the car and ends
   the session as aborted, completed or collision.

A failed chat completion is retried once; a second failure stalls the turn until the operator
sends a message, while the driver keeps its last brief.

## Google Gemini direct

A chat model `gemini/<exact model ID>` calls Google's Gemini API (its OpenAI-compatible endpoint)
with the `drivingbench-gemini` key; `google/<model ID>` stays on OpenRouter. Normal requests
Google's standard tier and fast its priority tier; there is no ultrafast. **Refresh Google models**
lists the catalog, which does not establish access or quota; **Other Google model…** takes an
exact ID. **Test Gemini connection** makes two real inference calls with a synthetic image and the
public tool definitions, checking a tool call, its result, image input and thought signatures,
without contacting the car. Thought signatures are kept verbatim in the history; the served tier is
recorded only when Google reports it, and the UI says "billing unavailable" rather than $0.

## Trained policies

The operator lists policy servers in `~/.config/drivingbench-v01/policies.json` (the config's
`policies` key moves it); nothing is committed. The gateway reads it when it starts, so restart it
after editing (`launchctl kickstart -k gui/$(id -u)/com.drivingbench.v01.gateway`), and refuses to
start with an invalid entry (`drivingbench serve` prints why).

```json
{
  "cones": {"url": "http://127.0.0.1:8000/act", "objective": "Follow the cones."},
  "cones-steer": {
    "url": "http://127.0.0.1:8001/act",
    "cameras": ["road_left", "road_right"],
    "steerable": true
  }
}
```

`url` is required; `cameras` defaults to `["front"]`; `objective` is the instruction the policy was
trained on; `steerable` defaults to false. Unknown fields are refused, and a name may not be `none`
or a decision model's ID. The registry holds no credentials: reach a remote server over a private
network or an SSH tunnel.

### The HTTP contract

About three times a second, one request at a time, the gateway POSTs JSON and waits up to 10 s:

```json
{"images": {"front": "<base64 JPEG>"}, "state": [0.012, 1.4], "instruction": "Follow the cones."}
```

- `images`: one RGB JPEG (quality 90) per configured view. `front` and `wide` are the narrow and
  wide road cameras' 1344×760 frames, with the shared image adjustments applied, resized
  (bilinear) to 672×380. `road_left` and `road_right` are the left and right halves of the
  672×380 narrow frame below row 140, 336×240 each.
- `state`: the measured curvature (1/m, left positive) and speed (m/s).
- `instruction`: the objective, or for a steerable policy the current command.

The reply is `{"actions": [[curvature_inv_m, speed_mps], ...]}`: rows 0.1 s apart, row 0 at the
older frame's capture. The comma runs the row due at each 0.1 s step on its own clock, 0.2 s early
for the steering's response delay, so time spent in inference and transit is skipped, not
replayed. The first 60 rows are used. Curvature is clipped to ±1 /m and capped by the shared
steering scale; a speed below 0.15 m/s brakes, others run at least 0.5 m/s and at most the shared
ceiling. A plan shorter than 0.8 s holds its last row to 0.8 s, and one that ended less than 1 s
before it arrived holds its last row; a later one brakes. Without a new plan, the car brakes at the
plan's end.

An HTTP error, a timeout, invalid JSON, a missing `actions`, or anything but a non-empty list of
two finite numbers per row sends nothing for that tick: the running chunk ends on its own, and the
error appears in the driver's decisions and the run log. At Start, one request with black views
and a zero state checks the endpoint.

A minimal server that crawls straight ahead:

```python
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Policy(BaseHTTPRequestHandler):
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        actions = [[0.0, 0.6]] * 20  # predict from request["images"], ["state"], ["instruction"]
        body = json.dumps({"actions": actions}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


HTTPServer(("127.0.0.1", 8000), Policy).serve_forever()
```

### Steerable policies

A steerable policy is trained with a per-frame instruction from seven commands, each true of the
car's own next 1.5 s: `stop` if it ends below 0.15 m/s; `sharp left`/`sharp right` if its mean
curvature exceeds 1/12 m⁻¹ (radius under 12 m); `curve left`/`curve right` above 1/50 m⁻¹;
`slow down` if it ends below three quarters of its current speed; otherwise `go straight`. It starts
on `go straight`; a supervising chat model chooses the next command with `set_brief`, and any other
text is refused with `not_a_command`.

## What is recorded

Each run logs both layers, every call, to `<runs>/runner/<run>/events.jsonl` (the config's `runs`,
by default `~/.local/share/drivingbench-v01/runs`): the configuration, session and attempt
markers, each driver frame's IDs, capture times and ages, each decision with its latency, command,
result and evidence (skipped and held steps for a policy), driver errors, briefs and pauses, and the
chat model's messages, responses (latency, usage, provider, model, requested and served tier) and
tool calls, with the run's cost when the provider reports it. The comma records every command in
the segment, as for any client; `drivingbench sync` attaches the run log to the segments it drove
([traces/README.md](../traces/README.md)).
