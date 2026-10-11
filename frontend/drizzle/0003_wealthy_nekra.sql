CREATE TABLE "safety_quizzes" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"session_id" text NOT NULL,
	"quiz_id" text NOT NULL,
	"visitor_id" text NOT NULL,
	"ticker" text NOT NULL,
	"stock_name" text,
	"questions" jsonb NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE INDEX "safety_quizzes_session_id_idx" ON "safety_quizzes" USING btree ("session_id");--> statement-breakpoint
CREATE INDEX "safety_quizzes_quiz_id_idx" ON "safety_quizzes" USING btree ("quiz_id");--> statement-breakpoint
CREATE INDEX "safety_quizzes_visitor_id_idx" ON "safety_quizzes" USING btree ("visitor_id");