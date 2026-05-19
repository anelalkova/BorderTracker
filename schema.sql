--
-- PostgreSQL database dump
--

\restrict 9xJmtTA8thpERvb7dXaFaaU4icIH5g4rb88xX34pHsudeTNGCXQ67XeC2wrvkTA

-- Dumped from database version 18.3
-- Dumped by pg_dump version 18.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: crossing_queue_multipliers; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.crossing_queue_multipliers (
    id integer NOT NULL,
    crossing_id integer NOT NULL,
    multiplier real NOT NULL,
    confidence text NOT NULL,
    matched_pairs integer NOT NULL,
    computed_at timestamp with time zone DEFAULT now() NOT NULL,
    notes text
);


ALTER TABLE public.crossing_queue_multipliers OWNER TO postgres;

--
-- Name: crossing_queue_multipliers_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.crossing_queue_multipliers_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.crossing_queue_multipliers_id_seq OWNER TO postgres;

--
-- Name: crossing_queue_multipliers_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.crossing_queue_multipliers_id_seq OWNED BY public.crossing_queue_multipliers.id;


--
-- Name: crossings; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.crossings (
    id integer NOT NULL,
    name text NOT NULL,
    display_name text NOT NULL,
    neighbor text NOT NULL
);


ALTER TABLE public.crossings OWNER TO postgres;

--
-- Name: crossings_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.crossings_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.crossings_id_seq OWNER TO postgres;

--
-- Name: crossings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.crossings_id_seq OWNED BY public.crossings.id;


--
-- Name: crowdsourced_waits; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.crowdsourced_waits (
    id integer NOT NULL,
    quality_flag text,
    camera_avg_min real,
    crossing_id integer,
    reported_at timestamp with time zone,
    wait_minutes integer
);


ALTER TABLE public.crowdsourced_waits OWNER TO postgres;

--
-- Name: crowdsourced_waits_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.crowdsourced_waits_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.crowdsourced_waits_id_seq OWNER TO postgres;

--
-- Name: crowdsourced_waits_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.crowdsourced_waits_id_seq OWNED BY public.crowdsourced_waits.id;


--
-- Name: snapshots; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.snapshots (
    id bigint NOT NULL,
    crossing_id integer NOT NULL,
    captured_at timestamp with time zone NOT NULL,
    interval_minutes integer NOT NULL,
    total_vehicles integer NOT NULL,
    cars integer DEFAULT 0 NOT NULL,
    motorcycles integer DEFAULT 0 NOT NULL,
    buses integer DEFAULT 0 NOT NULL,
    trucks integer DEFAULT 0 NOT NULL,
    lane_breakdown jsonb,
    fps real
);


ALTER TABLE public.snapshots OWNER TO postgres;

--
-- Name: snapshots_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.snapshots_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.snapshots_id_seq OWNER TO postgres;

--
-- Name: snapshots_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.snapshots_id_seq OWNED BY public.snapshots.id;


--
-- Name: vehicle_crossings; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vehicle_crossings (
    id bigint NOT NULL,
    crossing_id integer NOT NULL,
    track_id integer NOT NULL,
    vehicle_type text,
    lane text,
    entered_at timestamp with time zone NOT NULL,
    exited_at timestamp with time zone,
    duration_sec real,
    was_reassigned boolean DEFAULT false,
    frame_count integer DEFAULT 0,
    avg_confidence real,
    notes text
);


ALTER TABLE public.vehicle_crossings OWNER TO postgres;

--
-- Name: v_avg_crossing_times; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_avg_crossing_times AS
 SELECT c.name AS crossing,
    vc.lane,
    date_trunc('hour'::text, vc.entered_at) AS hour_utc,
    count(*) AS vehicles,
    round((avg(vc.duration_sec))::numeric, 1) AS avg_duration_sec,
    round((min(vc.duration_sec))::numeric, 1) AS min_duration_sec,
    round((max(vc.duration_sec))::numeric, 1) AS max_duration_sec,
    round((avg(vc.avg_confidence))::numeric, 3) AS avg_detection_confidence
   FROM (public.vehicle_crossings vc
     JOIN public.crossings c ON ((vc.crossing_id = c.id)))
  WHERE ((vc.duration_sec > (10)::double precision) AND (vc.duration_sec < (7200)::double precision) AND (vc.exited_at IS NOT NULL))
  GROUP BY c.name, vc.lane, (date_trunc('hour'::text, vc.entered_at));


ALTER VIEW public.v_avg_crossing_times OWNER TO postgres;

--
-- Name: wait_time_estimates; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.wait_time_estimates (
    id bigint NOT NULL,
    crossing_id integer NOT NULL,
    estimated_at timestamp with time zone NOT NULL,
    estimated_wait_minutes real,
    confidence real,
    model_version text,
    context_json jsonb
);


ALTER TABLE public.wait_time_estimates OWNER TO postgres;

--
-- Name: v_latest_estimates; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_latest_estimates AS
 SELECT DISTINCT ON (e.crossing_id) c.name,
    c.display_name,
    e.estimated_at,
    e.estimated_wait_minutes,
    e.confidence,
    e.model_version
   FROM (public.wait_time_estimates e
     JOIN public.crossings c ON ((e.crossing_id = c.id)))
  ORDER BY e.crossing_id, e.estimated_at DESC;


ALTER VIEW public.v_latest_estimates OWNER TO postgres;

--
-- Name: v_latest_snapshots; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_latest_snapshots AS
 SELECT DISTINCT ON (s.crossing_id) c.name,
    c.display_name,
    c.neighbor,
    s.captured_at,
    s.total_vehicles,
    s.cars,
    s.motorcycles,
    s.buses,
    s.trucks
   FROM (public.snapshots s
     JOIN public.crossings c ON ((s.crossing_id = c.id)))
  ORDER BY s.crossing_id, s.captured_at DESC;


ALTER VIEW public.v_latest_snapshots OWNER TO postgres;

--
-- Name: v_current_status; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_current_status AS
 SELECT ls.name,
    ls.display_name,
    ls.neighbor,
    ls.captured_at AS last_snapshot_at,
    ls.total_vehicles AS current_queue,
    ls.cars,
    ls.buses,
    ls.trucks,
    le.estimated_wait_minutes AS last_estimated_wait,
    le.confidence AS last_confidence,
    le.estimated_at AS last_estimated_at,
    ct.avg_duration_sec AS recent_avg_crossing_sec,
    ct.vehicles AS vehicles_tracked_this_hour
   FROM ((public.v_latest_snapshots ls
     LEFT JOIN public.v_latest_estimates le ON ((ls.name = le.name)))
     LEFT JOIN public.v_avg_crossing_times ct ON (((ls.name = ct.crossing) AND (ct.hour_utc = date_trunc('hour'::text, now())))));


ALTER VIEW public.v_current_status OWNER TO postgres;

--
-- Name: v_hourly_averages; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_hourly_averages AS
 SELECT c.name AS crossing,
    date_trunc('hour'::text, s.captured_at) AS hour_utc,
    count(*) AS snapshots,
    round(avg(s.total_vehicles), 1) AS avg_vehicles,
    max(s.total_vehicles) AS peak_vehicles,
    round(avg(s.cars), 1) AS avg_cars,
    round(avg(s.buses), 1) AS avg_buses,
    round(avg(s.trucks), 1) AS avg_trucks
   FROM (public.snapshots s
     JOIN public.crossings c ON ((s.crossing_id = c.id)))
  GROUP BY c.name, (date_trunc('hour'::text, s.captured_at));


ALTER VIEW public.v_hourly_averages OWNER TO postgres;

--
-- Name: v_throughput; Type: VIEW; Schema: public; Owner: postgres
--

CREATE VIEW public.v_throughput AS
 SELECT c.name AS crossing,
    date_trunc('hour'::text, vc.entered_at) AS hour_utc,
    count(*) AS vehicles_completed,
    round((avg(vc.duration_sec))::numeric, 1) AS avg_duration_sec,
    round((avg((vc.duration_sec / (60.0)::double precision)))::numeric, 2) AS avg_duration_min
   FROM (public.vehicle_crossings vc
     JOIN public.crossings c ON ((vc.crossing_id = c.id)))
  WHERE ((vc.duration_sec > (10)::double precision) AND (vc.duration_sec < (7200)::double precision) AND (vc.exited_at IS NOT NULL))
  GROUP BY c.name, (date_trunc('hour'::text, vc.entered_at));


ALTER VIEW public.v_throughput OWNER TO postgres;

--
-- Name: vehicle_crossings_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.vehicle_crossings_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.vehicle_crossings_id_seq OWNER TO postgres;

--
-- Name: vehicle_crossings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.vehicle_crossings_id_seq OWNED BY public.vehicle_crossings.id;


--
-- Name: wait_time_estimates_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.wait_time_estimates_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.wait_time_estimates_id_seq OWNER TO postgres;

--
-- Name: wait_time_estimates_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.wait_time_estimates_id_seq OWNED BY public.wait_time_estimates.id;


--
-- Name: crossing_queue_multipliers id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.crossing_queue_multipliers ALTER COLUMN id SET DEFAULT nextval('public.crossing_queue_multipliers_id_seq'::regclass);


--
-- Name: crossings id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.crossings ALTER COLUMN id SET DEFAULT nextval('public.crossings_id_seq'::regclass);


--
-- Name: crowdsourced_waits id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.crowdsourced_waits ALTER COLUMN id SET DEFAULT nextval('public.crowdsourced_waits_id_seq'::regclass);


--
-- Name: snapshots id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.snapshots ALTER COLUMN id SET DEFAULT nextval('public.snapshots_id_seq'::regclass);


--
-- Name: vehicle_crossings id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vehicle_crossings ALTER COLUMN id SET DEFAULT nextval('public.vehicle_crossings_id_seq'::regclass);


--
-- Name: wait_time_estimates id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.wait_time_estimates ALTER COLUMN id SET DEFAULT nextval('public.wait_time_estimates_id_seq'::regclass);


--
-- Data for Name: crossing_queue_multipliers; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.crossing_queue_multipliers (id, crossing_id, multiplier, confidence, matched_pairs, computed_at, notes) FROM stdin;
\.


--
-- Data for Name: crossings; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.crossings (id, name, display_name, neighbor) FROM stdin;
1	bogorodica	Bogorodica (МК–ГР)	Greece
2	blace	Blace (МК–КС)	Kosovo
3	tabanovce	Tabanovce (МК–СР)	Serbia
4	deve_bair	Deve Bair (МК–БГ)	Bulgaria
5	kafasan	Kafasan (МК–АЛ)	Albania
6	medzitlija	Megjitlija (МК–ГР)	Greece
\.


--
-- Data for Name: crowdsourced_waits; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.crowdsourced_waits (id, quality_flag, camera_avg_min, crossing_id, reported_at, wait_minutes) FROM stdin;
\.


--
-- Data for Name: snapshots; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.snapshots (id, crossing_id, captured_at, interval_minutes, total_vehicles, cars, motorcycles, buses, trucks, lane_breakdown, fps) FROM stdin;
1	4	2026-05-16 16:27:35.551917+02	5	3	2	0	0	1	{"DeveBair L1": {"total": 1, "by_type": {"truck": 1}}, "DeveBair L2": {"total": 2, "by_type": {"car": 2}}}	2.5
\.


--
-- Data for Name: vehicle_crossings; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vehicle_crossings (id, crossing_id, track_id, vehicle_type, lane, entered_at, exited_at, duration_sec, was_reassigned, frame_count, avg_confidence, notes) FROM stdin;
1	4	1	bus	DeveBair L1	2026-05-07 01:00:42.094129+02	2026-05-07 01:01:36.646508+02	54.55	f	145	0.724	\N
2	3	2	car	\N	2026-05-07 01:04:32.08928+02	2026-05-07 01:04:40.146935+02	8.06	f	23	0.393	\N
3	3	5	car	\N	2026-05-07 01:04:24.024717+02	2026-05-07 01:04:40.146935+02	16.12	f	45	0.446	\N
4	3	4	car	\N	2026-05-07 01:04:18.914602+02	2026-05-07 01:04:42.767041+02	23.85	f	66	0.43	\N
5	3	4	car	\N	2026-05-07 01:04:46.054163+02	2026-05-07 01:04:54.068244+02	8.01	f	23	0.473	\N
6	3	2	car	\N	2026-05-07 01:04:51.515283+02	2026-05-07 01:05:02.283656+02	10.77	f	30	0.346	\N
7	3	5	car	\N	2026-05-07 01:04:54.068244+02	2026-05-07 01:05:02.677382+02	8.61	f	24	0.418	\N
8	3	5	car	\N	2026-05-07 01:05:04.524844+02	2026-05-07 01:05:13.518585+02	8.99	f	25	0.363	\N
9	3	4	car	\N	2026-05-07 01:04:57.085929+02	2026-05-07 01:05:16.139348+02	19.05	f	52	0.404	\N
10	3	4	truck	\N	2026-05-07 01:05:16.932929+02	2026-05-07 01:05:29.792766+02	12.86	f	36	0.41	\N
11	3	29	car	\N	2026-05-07 01:05:25.355402+02	2026-05-07 01:05:30.57347+02	5.22	f	15	0.336	\N
12	3	4	car	\N	2026-05-07 01:05:33.142974+02	2026-05-07 01:05:44.640368+02	11.5	f	32	0.405	\N
13	3	31	car	\N	2026-05-07 01:05:36.052737+02	2026-05-07 01:05:46.522973+02	10.47	f	29	0.407	\N
14	3	1	truck	Tabanovce L1	2026-05-07 01:04:12.735053+02	2026-05-07 01:05:57.022112+02	104.29	f	283	0.81	\N
15	3	30	car	\N	2026-05-07 01:05:55.489131+02	2026-05-07 01:06:01.127509+02	5.64	f	16	0.418	\N
16	3	30	car	\N	2026-05-07 01:06:03.366582+02	2026-05-07 01:06:11.110407+02	7.74	f	22	0.544	\N
17	3	52	car	Tabanovce L3	2026-05-07 01:06:01.509103+02	2026-05-07 01:06:12.56164+02	11.05	f	31	0.61	\N
18	3	31	car	\N	2026-05-07 01:06:12.199106+02	2026-05-07 01:06:17.300864+02	5.1	f	15	0.416	\N
19	3	30	car	\N	2026-05-07 01:06:14.738338+02	2026-05-07 01:06:21.414331+02	6.68	f	19	0.537	\N
20	4	8	car	\N	2026-05-16 16:22:39.279796+02	2026-05-16 16:23:10.36277+02	31.08	f	68	0.575	\N
21	4	15	car	\N	2026-05-16 16:23:03.153636+02	2026-05-16 16:23:10.36277+02	7.21	f	19	0.494	\N
22	4	21	car	\N	2026-05-16 16:23:13.741098+02	2026-05-16 16:23:25.933263+02	12.19	f	26	0.499	\N
23	4	22	car	\N	2026-05-16 16:23:27.729636+02	2026-05-16 16:23:47.580111+02	19.85	f	37	0.543	\N
24	4	2	car	DeveBair L2	2026-05-16 16:22:39.279796+02	2026-05-16 16:23:56.396676+02	77.12	f	158	0.908	\N
25	4	27	car	\N	2026-05-16 16:24:05.095835+02	2026-05-16 16:24:16.147549+02	11.05	f	29	0.689	\N
26	4	7	car	DeveBair L2	2026-05-16 16:22:39.279796+02	2026-05-16 16:24:16.588276+02	97.31	f	206	0.626	\N
27	4	8	car	\N	2026-05-16 16:23:11.168633+02	2026-05-16 16:24:20.478302+02	69.31	f	146	0.511	\N
28	4	28	truck	\N	2026-05-16 16:24:05.485613+02	2026-05-16 16:24:20.478302+02	14.99	f	38	0.563	\N
29	4	8	truck	\N	2026-05-16 16:24:21.245004+02	2026-05-16 16:24:29.965145+02	8.72	f	22	0.628	\N
30	4	27	car	\N	2026-05-16 16:24:24.153662+02	2026-05-16 16:24:30.513009+02	6.36	f	16	0.373	\N
31	4	3	car	DeveBair L2	2026-05-16 16:22:39.279796+02	2026-05-16 16:24:58.66738+02	139.39	f	309	0.916	\N
32	4	36	car	\N	2026-05-16 16:24:31.426468+02	2026-05-16 16:25:23.869515+02	52.44	f	103	0.529	\N
33	4	8	truck	\N	2026-05-16 16:24:30.948794+02	2026-05-16 16:26:00.125331+02	89.18	f	186	0.552	\N
34	4	48	car	\N	2026-05-16 16:25:25.523192+02	2026-05-16 16:26:00.125331+02	34.6	f	80	0.536	\N
35	4	8	car	\N	2026-05-16 16:26:01.066953+02	2026-05-16 16:26:21.363356+02	20.3	f	47	0.549	\N
36	4	4	car	DeveBair L2	2026-05-16 16:22:39.279796+02	2026-05-16 16:26:42.431423+02	243.15	f	521	0.869	\N
37	4	54	truck	\N	2026-05-16 16:26:32.871004+02	2026-05-16 16:26:55.064774+02	22.19	f	52	0.53	\N
38	4	8	truck	\N	2026-05-16 16:26:22.653279+02	2026-05-16 16:27:00.303447+02	37.65	f	87	0.587	\N
39	4	8	car	\N	2026-05-16 16:27:01.220088+02	2026-05-16 16:27:16.589784+02	15.37	f	38	0.617	\N
40	4	7	truck	DeveBair L2	2026-05-16 16:24:17.558465+02	2026-05-16 16:27:27.968106+02	190.41	f	418	0.768	\N
41	4	8	car	\N	2026-05-16 16:27:17.641908+02	2026-05-16 16:27:30.4285+02	12.79	f	30	0.525	\N
42	4	80	car	\N	2026-05-16 16:27:20.915488+02	2026-05-16 16:27:30.4285+02	9.51	f	23	0.589	\N
43	4	84	car	\N	2026-05-16 16:27:31.669357+02	2026-05-16 16:27:39.611452+02	7.94	f	19	0.581	\N
44	4	8	truck	\N	2026-05-16 16:27:31.265241+02	2026-05-16 16:28:03.65563+02	32.39	f	77	0.627	\N
45	4	90	car	\N	2026-05-16 16:27:51.689967+02	2026-05-16 16:28:03.65563+02	11.97	f	30	0.501	\N
46	4	8	car	\N	2026-05-16 16:28:04.425647+02	2026-05-16 16:28:21.984534+02	17.56	f	40	0.49	\N
47	4	94	truck	\N	2026-05-16 16:28:04.803592+02	2026-05-16 16:28:21.984534+02	17.18	f	39	0.607	\N
48	4	27	car	DeveBair L2	2026-05-16 16:24:33.86595+02	2026-05-16 16:28:55.795598+02	261.93	f	524	0.781	\N
49	4	95	car	\N	2026-05-16 16:28:23.986067+02	2026-05-16 16:29:01.946015+02	37.96	f	30	0.515	\N
50	4	8	truck	\N	2026-05-16 16:28:22.801084+02	2026-05-16 16:29:07.619008+02	44.82	f	47	0.579	\N
51	4	6	truck	\N	2026-05-16 16:22:39.279796+02	2026-05-16 16:29:19.256461+02	399.98	f	826	0.551	\N
52	4	6	truck	\N	2026-05-16 16:29:20.419887+02	2026-05-16 16:29:29.911448+02	9.49	f	23	0.489	\N
53	4	3	car	DeveBair L2	2026-05-16 16:49:58.022304+02	2026-05-16 16:50:12.854186+02	14.83	f	37	0.628	\N
54	4	6	car	\N	2026-05-16 16:49:54.441774+02	2026-05-16 16:50:24.815519+02	30.37	f	76	0.558	\N
55	4	6	car	\N	2026-05-16 16:50:25.593917+02	2026-05-16 16:50:37.038251+02	11.44	f	29	0.549	\N
56	4	3	truck	DeveBair L2	2026-05-16 16:50:31.760932+02	2026-05-16 16:50:38.342757+02	6.58	f	17	0.616	\N
57	4	50	truck	DeveBair L2	2026-05-16 16:50:34.642645+02	2026-05-16 16:50:48.076769+02	13.43	f	31	0.589	\N
58	4	55	truck	\N	2026-05-16 16:50:40.704502+02	2026-05-16 16:50:56.396457+02	15.69	f	29	0.508	\N
59	4	3	truck	DeveBair L2	2026-05-16 16:50:39.126652+02	2026-05-16 16:50:57.579521+02	18.45	f	36	0.667	\N
60	4	66	truck	DeveBair L2	2026-05-16 16:51:00.318348+02	2026-05-16 16:51:13.411959+02	13.09	f	15	0.607	\N
61	4	3	car	DeveBair L2	2026-05-16 16:50:58.360532+02	2026-05-16 16:51:13.411959+02	15.05	f	19	0.711	\N
62	4	69	car	\N	2026-05-16 16:51:01.871624+02	2026-05-16 16:51:19.957941+02	18.09	f	18	0.509	\N
63	4	1	truck	\N	2026-05-16 16:49:54.441774+02	2026-05-16 16:51:20.764827+02	86.32	f	175	0.879	\N
64	4	2	car	DeveBair L2	2026-05-16 16:49:54.441774+02	2026-05-16 16:51:25.761992+02	91.32	f	187	0.754	\N
65	4	89	truck	DeveBair L2	2026-05-16 16:51:32.609948+02	2026-05-16 16:51:40.819661+02	8.21	f	22	0.635	\N
66	4	3	truck	DeveBair L2	2026-05-16 16:51:14.236796+02	2026-05-16 16:51:40.819661+02	26.58	f	58	0.69	\N
67	4	6	car	\N	2026-05-16 16:50:37.926038+02	2026-05-16 16:51:40.819661+02	62.89	f	118	0.542	\N
68	4	2	car	DeveBair L2	2026-05-16 16:51:26.598701+02	2026-05-16 16:51:53.360607+02	26.76	f	65	0.711	\N
69	4	103	truck	DeveBair L2	2026-05-16 16:51:45.459493+02	2026-05-16 16:51:53.360607+02	7.9	f	17	0.624	\N
70	4	6	car	\N	2026-05-16 16:51:41.590484+02	2026-05-16 16:51:54.14768+02	12.56	f	29	0.543	\N
71	4	2	truck	DeveBair L2	2026-05-16 16:51:54.14768+02	2026-05-16 16:52:00.675349+02	6.53	f	16	0.628	\N
72	4	109	truck	\N	2026-05-16 16:51:53.752107+02	2026-05-16 16:52:00.675349+02	6.92	f	17	0.43	\N
73	4	105	truck	DeveBair L2	2026-05-16 16:51:46.249092+02	2026-05-16 16:52:27.646162+02	41.4	f	55	0.594	\N
74	4	3	car	DeveBair L2	2026-05-16 16:51:41.590484+02	2026-05-16 16:52:27.646162+02	46.06	f	67	0.64	\N
75	4	123	truck	\N	2026-05-16 16:52:19.226764+02	2026-05-16 16:52:38.150683+02	18.92	f	39	0.544	\N
76	4	3	truck	DeveBair L2	2026-05-16 16:52:28.898092+02	2026-05-16 16:52:38.936085+02	10.04	f	26	0.709	\N
77	4	6	car	\N	2026-05-16 16:51:54.974054+02	2026-05-16 16:52:41.316222+02	46.34	f	70	0.481	\N
\.


--
-- Data for Name: wait_time_estimates; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.wait_time_estimates (id, crossing_id, estimated_at, estimated_wait_minutes, confidence, model_version, context_json) FROM stdin;
\.


--
-- Name: crossing_queue_multipliers_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.crossing_queue_multipliers_id_seq', 1, false);


--
-- Name: crossings_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.crossings_id_seq', 30, true);


--
-- Name: crowdsourced_waits_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.crowdsourced_waits_id_seq', 1, false);


--
-- Name: snapshots_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.snapshots_id_seq', 1, true);


--
-- Name: vehicle_crossings_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.vehicle_crossings_id_seq', 77, true);


--
-- Name: wait_time_estimates_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.wait_time_estimates_id_seq', 1, false);


--
-- Name: crossing_queue_multipliers crossing_queue_multipliers_crossing_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.crossing_queue_multipliers
    ADD CONSTRAINT crossing_queue_multipliers_crossing_id_key UNIQUE (crossing_id);


--
-- Name: crossing_queue_multipliers crossing_queue_multipliers_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.crossing_queue_multipliers
    ADD CONSTRAINT crossing_queue_multipliers_pkey PRIMARY KEY (id);


--
-- Name: crossings crossings_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.crossings
    ADD CONSTRAINT crossings_name_key UNIQUE (name);


--
-- Name: crossings crossings_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.crossings
    ADD CONSTRAINT crossings_pkey PRIMARY KEY (id);


--
-- Name: crowdsourced_waits crowdsourced_waits_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.crowdsourced_waits
    ADD CONSTRAINT crowdsourced_waits_pkey PRIMARY KEY (id);


--
-- Name: snapshots snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.snapshots
    ADD CONSTRAINT snapshots_pkey PRIMARY KEY (id);


--
-- Name: vehicle_crossings vehicle_crossings_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vehicle_crossings
    ADD CONSTRAINT vehicle_crossings_pkey PRIMARY KEY (id);


--
-- Name: wait_time_estimates wait_time_estimates_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.wait_time_estimates
    ADD CONSTRAINT wait_time_estimates_pkey PRIMARY KEY (id);


--
-- Name: idx_cw_crossing_reported; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_cw_crossing_reported ON public.crowdsourced_waits USING btree (crossing_id, reported_at);


--
-- Name: idx_estimates_crossing_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_estimates_crossing_time ON public.wait_time_estimates USING btree (crossing_id, estimated_at DESC);


--
-- Name: idx_snapshots_crossing_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_snapshots_crossing_time ON public.snapshots USING btree (crossing_id, captured_at DESC);


--
-- Name: idx_snapshots_lane_breakdown; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_snapshots_lane_breakdown ON public.snapshots USING gin (lane_breakdown);


--
-- Name: idx_snapshots_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_snapshots_time ON public.snapshots USING btree (captured_at DESC);


--
-- Name: idx_vc_crossing_entered; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_vc_crossing_entered ON public.vehicle_crossings USING btree (crossing_id, entered_at) WHERE ((exited_at IS NOT NULL) AND (duration_sec > (0)::double precision));


--
-- Name: idx_vehicle_crossings_crossing_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_vehicle_crossings_crossing_time ON public.vehicle_crossings USING btree (crossing_id, entered_at DESC);


--
-- Name: idx_vehicle_crossings_duration; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_vehicle_crossings_duration ON public.vehicle_crossings USING btree (duration_sec);


--
-- Name: idx_vehicle_crossings_lane; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_vehicle_crossings_lane ON public.vehicle_crossings USING btree (lane, entered_at DESC);


--
-- Name: crossing_queue_multipliers crossing_queue_multipliers_crossing_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.crossing_queue_multipliers
    ADD CONSTRAINT crossing_queue_multipliers_crossing_id_fkey FOREIGN KEY (crossing_id) REFERENCES public.crossings(id);


--
-- Name: snapshots snapshots_crossing_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.snapshots
    ADD CONSTRAINT snapshots_crossing_id_fkey FOREIGN KEY (crossing_id) REFERENCES public.crossings(id);


--
-- Name: vehicle_crossings vehicle_crossings_crossing_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vehicle_crossings
    ADD CONSTRAINT vehicle_crossings_crossing_id_fkey FOREIGN KEY (crossing_id) REFERENCES public.crossings(id);


--
-- Name: wait_time_estimates wait_time_estimates_crossing_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.wait_time_estimates
    ADD CONSTRAINT wait_time_estimates_crossing_id_fkey FOREIGN KEY (crossing_id) REFERENCES public.crossings(id);


--
-- PostgreSQL database dump complete
--

\unrestrict 9xJmtTA8thpERvb7dXaFaaU4icIH5g4rb88xX34pHsudeTNGCXQ67XeC2wrvkTA

